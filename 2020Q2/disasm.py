import os
import subprocess
import textwrap

FILE = '/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch_cuda.so'

def extract(cubin):
    cmd = f'rm -rf {cubin}; cuobjdump -xelf {cubin} {FILE}'
    print('Running:')
    print(cmd)
    print()
    os.system(cmd)

def demangle(symbol):
    out = subprocess.Popen(['c++filt', symbol],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    return stdout.decode().strip()

def get_asm(cubin):
    out = subprocess.Popen(['nvdisasm', '-c', '-g', '-ndf', cubin],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    lines = stdout.decode().split('\n')
    symbol = None
    codes = { None: [] }
    for l in lines:
        if len(l.strip()) > 0:
            if l.startswith('//-'):
                try:
                    symbol = l.split()[1].split('.')[2]
                    codes[symbol] = []
                except IndexError:
                    pass
            else:
                codes[symbol].append(l)
    for k, v in codes.items():
        codes[k] = textwrap.dedent('\n' + '\n'.join(v))
    del codes[None]
    return codes

def run(cubin, filter_):
    import torch
    print(torch.__version__)
    print(torch.version.git_version)
    print()
    extract(cubin)
    for s, c in get_asm(cubin).items():
        demangled = demangle(s)
        if filter_(demangled, c):
            print('**Symbol:**')
            print(demangled)
            print()
            print("**ASM:**")
            print(c)
            print()
