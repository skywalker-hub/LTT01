import types
from transformers.trainer import *


def patch_trainer_optimizer(trainer, lr_thinking_residual_gate=1e-4, thinking_residual_Lambda=1e-3, lr_thinking_residual_head=1e-4, lr_token_gate_matrix=1e-4):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            
            # ============ Debug: print all param names ============
            print("\n[patch.py DEBUG] Checking parameter names...")
            gate_params = []
            head_params = []
            for n, p in opt_model.named_parameters():
                if "token_gate" in n:
                    gate_params.append((n, p.requires_grad, p.numel()))
                if "thinking_residual_head" in n:
                    head_params.append((n, p.requires_grad, p.numel()))
            
            print(f"  token_gate_matrix 参数:")
            for n, req_grad, numel in gate_params:
                print(f"    {n} | requires_grad={req_grad} | numel={numel:,}")
            if not gate_params:
                print("    ⚠️ 未找到任何 token_gate 参数！")
            
            print(f"  thinking_residual_head 参数:")
            for n, req_grad, numel in head_params:
                print(f"    {n} | requires_grad={req_grad} | numel={numel:,}")
            if not head_params:
                print("    ⚠️ 未找到任何 thinking_residual_head 参数！")
            print("=" * 60)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("thinking_residual" not in n and "token_gate_matrix" not in n and n in decay_parameters and p.requires_grad)
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("thinking_residual" not in n and "token_gate_matrix" not in n and n not in decay_parameters and p.requires_grad)
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("thinking_residual_gate" in n and p.requires_grad)
                    ],
                    "lr": lr_thinking_residual_gate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("thinking_residual_Lambda" in n and p.requires_grad)
                    ],
                    "lr": thinking_residual_Lambda,
                    "weight_decay": self.args.weight_decay,
                },
                # 新增: thinking_residual_head 参数组，学习率与门控矩阵相同
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("thinking_residual_head" in n and p.requires_grad)
                    ],
                    "lr": lr_thinking_residual_head,
                    "weight_decay": self.args.weight_decay,
                },
                # 新增: token_gate_matrix 参数组
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("token_gate_matrix" in n and p.requires_grad)
                    ],
                    "lr": lr_token_gate_matrix,
                    "weight_decay": self.args.weight_decay,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    trainer._old_create_optimizer = trainer.create_optimizer
    trainer.create_optimizer = types.MethodType(create_optimizer, trainer)