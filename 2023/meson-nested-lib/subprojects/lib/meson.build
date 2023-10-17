project('hello-lib', 'cpp')

lib = library('lib', 'lib.cpp')
backend = library('lib-backend', 'lib-backend.cpp', link_with:[lib])

lib_dep = declare_dependency(include_directories: ['.'], link_with: [lib])
backend_dep = declare_dependency(link_with: [backend])
