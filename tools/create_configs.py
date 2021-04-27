import os

def replace(filepath, old_str, new_str):
    with open(filepath, 'r') as fr, open(f'{filepath}.tmp', 'w') as fw:
        for line in fr:
            if old_str in line:
                line = line.replace(old_str, new_str)
            fw.write(line)
        os.remove(filepath)
        os.rename(f'{filepath}.tmp', filepath)

def find_all_files(path):
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            fullpath = os.path.join(dirpath, name)
            files.append(fullpath)
    return files

def main():
    files = find_all_files('./configs/zzp/baseline/baseline/')
    print(files)
    for filepath in files:
        replace(filepath, '../../../', '../../../../')
        replace(filepath, 'work_dir/baseline/', 'work_dir/baseline/baseline/')

if __name__ == '__main__':
    main()