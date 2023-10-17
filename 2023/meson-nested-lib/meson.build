project('hello', 'cpp')

lib_project = subproject('lib')
lib_dep = lib_project.get_variable('lib_dep')
backend_dep = lib_project.get_variable('backend_dep')

executable('hello', 'hello.cpp', dependencies: [lib_dep, backend_dep])
