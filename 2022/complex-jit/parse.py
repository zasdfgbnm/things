import clang.cindex

index = clang.cindex.Index.create()
tu = index.parse("a.cpp", unsaved_files=[("a.cpp", "#include <complex>")])
tu.save()
for n in tu.cursor.get_children():
    print(n)
    print(n.spelling)
