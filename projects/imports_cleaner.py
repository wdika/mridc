# coding=utf-8
# given a project folder find all possible circular imports and suggest a fix
#

import os
import sys


def get_all_files(project_dir):
    all_files = []
    for root, dirs, files in os.walk(project_dir):
        if "__pycache__" in root:
            continue
        for dir in dirs:
            if dir.startswith("__"):
                # print(f"Found directory {dir} in {root} (skipping it)")
                continue
            for file in files:
                if file.endswith(".py") and not file.endswith(".pyc"):
                    all_files.append(os.path.join(root, file))
        for file in files:
            if file.endswith(".py") and not file.endswith(".pyc"):
                all_files.append(os.path.join(root, file))
    return all_files


def get_imports(file):
    imports = []
    with open(file, "r") as f:
        for line in f.readlines():
            if line.startswith("from") or line.startswith("import"):
                imports.append(line)
    return imports


def get_imported_modules(imports):
    modules = []
    for line in imports:
        if line.startswith("from"):
            modules.append(line.split("from")[1].split("import")[0].strip())
        elif line.startswith("import"):
            modules.append(line.split("import")[1].strip())
    return modules


def get_imported_modules_from_file(file):
    imports = get_imports(file)
    return get_imported_modules(imports)


def get_imported_modules_from_files(files):
    all_modules = []
    for file in files:
        all_modules.extend(get_imported_modules_from_file(file))
    return all_modules


def get_imported_modules_from_project(project_dir):
    files = get_all_files(project_dir)
    return get_imported_modules_from_files(files)


def main(project_dir):
    modules = get_imported_modules_from_project(project_dir)
    # get only modules containing mridc
    modules = [module for module in modules if "mridc" in module and not "package_info" in module]

    # find circular imports
    circular_imports = {}
    files = get_all_files(project_dir)
    for module in modules:
        for file in files:
            if module in get_imported_modules_from_file(file):
                if module not in circular_imports:
                    circular_imports[module] = []
                circular_imports[module].append(file)

    # dump to json
    # import json
    # with open("circular_imports.json", "w") as f:
    #     json.dump(circular_imports, f, indent=4)


if __name__ == "__main__":
    main("mridc")
