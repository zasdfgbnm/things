add_executable(single_dynamic helloworld.cpp main.cpp)

add_executable(single_static helloworld.cpp main.cpp)
target_link_libraries(single_static -static)

add_library(hello SHARED helloworld.cpp)

add_library(hello_static STATIC helloworld.cpp)
# target_link_libraries(hello_static -static)

add_executable(multi_dynamic main.cpp)
target_link_libraries(multi_dynamic hello)

add_executable(multi_static main.cpp)
target_link_libraries(multi_static hello_static)
target_link_libraries(multi_static -static)
