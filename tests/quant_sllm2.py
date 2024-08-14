def _replace_with_custom_fn_if_matches_filter(
    model,
    replacement_fn,
    filter_fn,
    cur_fqn="",
) -> None:
    """
    Recursively replaces each child module in `model` with the result of `replacement_fn(child)`
    if `filter_fn(child)` returns `True`.

    Args:
        model (torch.nn.Module): The model containing modules to be replaced.
        replacement_fn (Callable[[torch.nn.Module], torch.nn.Module]): The function to replace matching modules.
        filter_fn (Callable[[torch.nn.Module], bool]): The filter function to determine which modules to replace.
        cur_fqn (str, optional): The current fully qualified name of the module being processed. Defaults to "".

    Returns:
        None
    """
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model)
        return model
    else:
        for name, child in model.named_children():
            new_child = _replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, f"{cur_fqn}{name}."
            )
            if new_child is not child:
                setattr(model, name, new_child)
        return model




import torch
from tempfile import TemporaryDirectory



with torch.no_grad():
    import transformers
    model_name = "codefuse-ai/CodeFuse-DeepSeek-33B"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype= 'auto')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    print(model)
    from accelerate import init_empty_weights, cpu_offload, disk_offload
    from accelerate.utils.timeit import TimeRecorder, Timer
    from accelerate.utils.offload import offload_state_dict_time_recorder
    def quant_linear(lin: torch.nn.Linear, bits: int = 8):
        """
                self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        
        """
        lin.qweight = torch.nn.Parameter(torch.zeros((lin.in_features, lin.out_features // (32 // bits)), dtype=torch.int32), requires_grad=False)
        del lin.weight
        return lin
    
    
    def quant_block_(block: torch.nn.Module, bits: int = 8):
        return _replace_with_custom_fn_if_matches_filter(
            block,
            replacement_fn=lambda x: quant_linear(x, bits=bits),
            filter_fn=lambda x, _: isinstance(x, torch.nn.Linear),
        )
    
    with TemporaryDirectory() as tempdir:
        total_quant = TimeRecorder("Total quantization time")
        total_offload = TimeRecorder("Total offload time")
        with Timer("Quantizing model "):
            for i, layer in enumerate(model.model.layers):
                print(f"Quantizing layer {i}")
                with Timer(f"Quantizing layer {i} ", total_quant):
                    quant_block_(layer, bits=4)
                with Timer(f"Offloading layer {i} ", total_offload):
                    disk_offload(layer, tempdir + f"/layer_{i}.pt", execution_device=torch.device("cpu"))
        print(total_quant)
        print(total_offload)
        print(offload_state_dict_time_recorder)
        
        
"""

Quantizing layer 61  time: 44 ms
Offload state dict time: 1351 ms
Offloading layer 61  time: 1366 ms
Quantizing model  time: 62450 ms
Total quantization time : total: 2786 ms, avg: 44.935483870967744 ms
Total offload time : total: 59655 ms, avg: 962.1774193548387 ms
Offload state dict time : total: 58735 ms, avg: 947.3387096774194 ms

w/ ThreadPoolExecutor

Offloading layer 61  time: 611 ms
Quantizing model  time: 47363 ms
Total quantization time : total: 4534 ms, avg: 73.12903225806451 ms
Total offload time : total: 42817 ms, avg: 690.5967741935484 ms
Offload state dict time : total: 41797 ms, avg: 674.1451612903226 ms
"""