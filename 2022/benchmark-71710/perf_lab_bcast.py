import torch
from matplotlib import pyplot as plt
from torch.utils.benchmark import Timer, Compare
import math
import click
#print(torch.cuda.get_device_capability()) # check that we are on Volta (compute capability 7,0)
#torch.cuda.set_device(1)
# don't benchmark on anything too small, you'll see only overhead
@click.command()
@click.option('--op_str', default="torch.gt")
@click.option('--dtype_str', default="float", type=click.Choice(['float', 'half', 'int', 'uint8', 'double']))
@click.option('--type_promotion', default=False)
def bench(op_str, dtype_str, type_promotion):
    device="cuda"
    def make_tensor(size, device, dtype):
        if type(size) == int:
            size = (size,)
        if dtype.is_floating_point:
            return torch.randn(size, device=device, dtype=dtype)
        else:
            return torch.randint(1,10, size, device=device, dtype=dtype)

    if dtype_str == "float":
        dtype = torch.float
    elif dtype_str == "half":
        dtype = torch.half
    elif dtype_str == "int":
        dtype = torch.int32
    elif dtype_str == "uint8":
        dtype = torch.uint8
    elif dtype_str == "double":
        dtype = torch.double

    MB = 1024 * 1024
    size = MB
    empty_input = torch.tensor([], dtype=dtype)
    element_size = empty_input.element_size()
    empty_output = eval(op_str+"(empty_input, empty_input)")
    output_element_size = empty_output.element_size()
    #print(element_size, output_element_size)
    sizes = []
    results = [[],[],[]]

    size = MB
    for _ in range(20):
        torch.cuda.memory.empty_cache()
        M = math.floor(math.sqrt(size))
        dtype_a = dtype_b = dtype
        if type_promotion:
            dtype_a = dtype_b = torch.float if dtype != torch.float else torch.half

        a=make_tensor((1, M), device=device, dtype=dtype_a)
        a_orig=make_tensor((1, M), device=device, dtype=dtype)
        b=make_tensor((M, M), device=device, dtype=dtype_b)
        b1 = make_tensor((M, 1), device=device, dtype=dtype)
        #dry runs
        out = eval(op_str + "(a_orig,b)")
        out = eval(op_str + "(a,b1)")
        out = eval(op_str + "(b,b1)")
        output_element_size = out.element_size()
        element_size_a = a.element_size()
        element_size_b = b.element_size()
        element_size_b1 = b1.element_size()

        t0 = Timer(stmt=f"{op_str}(a_orig,b)", label = op_str, sub_label=f"{M*M/MB} MB", description="MM1M", globals = {"a_orig":a_orig, "b":b})
        t1 = Timer(stmt=f"{op_str}(a,b1)", label = op_str, sub_label=f"{M*M/MB} MB", description="M11M", globals = {"a":a, "b1":b1})
        t2 = Timer(stmt=f"{op_str}(b,b1)", label = op_str, sub_label=f"{M*M/MB} MB", description="MMM1", globals = {"b":b, "b1":b1})

        #ts = Timer(stmt=f"{op_str}(b,1.)", label = op_str, sub_label=f"{M*M/MB} MB", description="scalar", globals = {"a":a, "b":b})

        res = [t.blocked_autorange() for t in (t0, t1, t2)]
        for (rl, r) in zip(results, res):
            rl.append(r)
        sizes.append(M)
        size += MB
        del a #to save memory for next iterations
        del b
    # comps = [Compare(r) for r in results]
    # for c in comps:
    #     print(c)
    bw=[[],[],[]]

    for res0, res1, res2, size in zip(results[0],results[1],results[2], sizes):
        bytes_io0 = size*element_size+size*size*element_size_b + output_element_size * size*size #(size+size+size*size)*4
        bytes_io1 = size*element_size_a + size*element_size_b1 + output_element_size * size*size #(size+size+size*size)*4
        bytes_io2 = size * element_size_b1 + size*size*element_size_b + output_element_size * size * size
        #print(output_element_size * size * size, bytes_io0, bytes_io1)
        bytes_iol = (bytes_io0, bytes_io1, bytes_io2)
        for (bw_elem, bytes_elem, res_elem) in zip(bw, bytes_iol, (res0, res1, res2)):
            bw_elem.append(bytes_elem/res_elem.median * 1e-9)
        print(f"{bytes_iol[0]/MB:7.1f} {bw[0][-1]:7.1f}", f"{bytes_iol[1]/MB:7.1f} {bw[1][-1]:7.1f}",
        f"{bytes_iol[2]/MB:7.1f} {bw[2][-1]:7.1f}")

if __name__ == '__main__':
    bench()
