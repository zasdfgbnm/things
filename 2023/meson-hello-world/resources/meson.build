stringify = find_program('../stringify.sh')
hello = custom_target(
    'hello',
    input: '../texts/hello.txt',
    output: 'hello.h',
    command: [stringify, '@INPUT@', '@OUTPUT@'],
)
