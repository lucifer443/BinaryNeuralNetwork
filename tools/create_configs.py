import os

def replace(filepath, old_str, new_str):
    with open(filepath, 'r') as fr, open(f'{filepath}.tmp', 'w') as fw:
        for line in fr:
            if old_str in line:
                line = line.replace(old_str, new_str)
            fw.write(line)
        filepath_split = filepath.split('/')
        name = filepath_split[-1].rstrip('.py')
        name_split = name.split('_')
        name_split[1:1] = ['prelu',]
        name = '_'.join(name_split)
        filepath_split[-1:] = [name + '.py',]
        filepath_new = '/'.join(filepath_split)
        print(filepath_new)
        os.remove(filepath)
        os.rename(f'{filepath}.tmp', filepath_new)

def find_all_files(path):
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            fullpath = os.path.join(dirpath, name)
            files.append(fullpath)
    return files

def main():
    files = find_all_files('./configs/zzp/baseline/baseline_prelu/')
    files = [f for f in files if 'prelu_2' in f]
    print(files)
    for filepath in files:
        replace(filepath, 'work_dir/baseline/baseline/baseline', 'work_dir/baseline/baseline_prelu/baseline_prelu')

if __name__ == '__main__':
    main()