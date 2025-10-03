from typing import Any, NamedTuple, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule, fully_shard

from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B
from torch.distributed.pipelining import PipelineStage

from torchtitan.tools.logging import logger


class AmdPipelineStage(PipelineStage):
    def backward_maybe_with_nosync(
        self,
        backward_type,
        bwd_kwargs: dict,
        last_backward: bool = False,
    ) -> tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]]:
        return super().backward_maybe_with_nosync(
            backward_type, bwd_kwargs, last_backward=False
        )

    def fsdp_pre_forward(self):
        module = self.submod
        distributed_state = fully_shard.state(module)  # type: ignore[attr-defined]

        for state in distributed_state._state_ctx.all_states:
            if state._fsdp_param_group:
                group = state._fsdp_param_group
                group.unshard(group.unshard_async_op)
                group.wait_for_unshard()


    def fsdp_post_backward(self):
        def run_post_backward(fsdp_module: FSDPModule) -> None:
            fsdp_module.set_is_last_backward(True)
            fsdp_module.set_reshard_after_backward(True)
            fsdp_module.set_requires_gradient_sync(True)

            distributed_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]

            for state in distributed_state._state_ctx.all_states:
                if state._fsdp_param_group:
                    state._fsdp_param_group.post_backward()

            # it would be much better if pipelining backward invoked .backward so autograd hooks
            # worked and modules like DDP/FSDP behaved as expected.  Working around this for the time being,
            # we need to call this too to ensure FSDP syncs its grad reduction ops back to the default stream.
            distributed_state._root_post_backward_final_callback()
        run_post_backward(self.submod)


class ScheduleAmdInterleaved1F1B(ScheduleInterleaved1F1B):
    def step(self, *args, target=None, losses: Optional[list] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
        if self._has_backward and not torch.is_grad_enabled():
            raise RuntimeError(
                "step() requires gradients to be enabled for backward computation; "
                "it should not be used under torch.no_grad() context. "
                "Please call eval() instead."
            )

        # Set the same has_backward flag for stage object
        for stage in self._stages:
            stage.has_backward = self._has_backward

        # Clean per iteration
        for stage in self._stages:
            stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        for stage in self._stages:
            stage.fsdp_pre_forward()
        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)
        for stage in self._stages:
            stage.fsdp_post_backward()

        # Return merged results per original format
        for stage in self._stages:
            if stage.is_last:
                return self._merge_outputs(stage.output_chunks)
        # Does not contain the last stage
        return None
