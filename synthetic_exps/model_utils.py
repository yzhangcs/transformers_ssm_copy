import fla
from models import (GPTNeoXAlibiForCausalLM, GPTNeoXHardAlibiForCausalLM,
                    GPTNeoXNoPEForCausalLM)
from transformers import (AutoConfig, AutoModelForCausalLM, GPTNeoXConfig,
                          GPTNeoXForCausalLM, PretrainedConfig)


def get_model(args, tokenizer):

    if args.model in ["T_nope", "T_rope", "T_alibi"]:
        config = GPTNeoXConfig(
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size*4,
            num_attention_heads=args.heads,
            num_hidden_layers=args.layers,
            vocab_size=len(tokenizer),
        )
    elif args.model == "T_hard_alibi":
        config = GPTNeoXConfig(
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size*4,
            num_attention_heads=args.heads,
            num_hidden_layers=args.layers,
            num_masked_heads=args.num_masked_heads,
            vocab_size=len(tokenizer),
        )

    if args.model == "T_rope":
        model = GPTNeoXForCausalLM(config)
    elif args.model == "T_nope":
        model = GPTNeoXNoPEForCausalLM(config)
    elif args.model == "T_alibi":
        model = GPTNeoXAlibiForCausalLM(config)
    elif args.model == "T_hard_alibi":
        model = GPTNeoXHardAlibiForCausalLM(config)
    else:
        config = {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.layers,
            "vocab_size": len(tokenizer),
        }
        if "mamba" in args.model:
            config["state_size"] = args.state_dim
        classes = [getattr(fla.models, i) for i in fla.models.__all__]
        configs = {i.model_type: i(**config) for i in classes if issubclass(i, PretrainedConfig)}
        config = configs[args.model] if args.model in configs else AutoConfig.from_pretrained(args.model)
        print(config)
        model = AutoModelForCausalLM.from_config(config)

    return model
