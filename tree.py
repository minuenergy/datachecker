import os

def print_tree(directory, indent=''):
    '''
    check your dataset's directory structure
    you can get this reult
        
    ├── /workspace/llm_dataset/ActivityNet/captions : 7  files,  {'.json', '.txt'}
        ├── /workspace/llm_dataset/ActivityNet/videos/train_video : 13329  files,  {'.mkv', '.webm', '.mp4', '.py'}
        ├── /workspace/llm_dataset/ActivityNet/videos/train_clip : 13213  files,  {'.pkl'}
        ├── /workspace/llm_dataset/ActivityNet/videos/all_test : 5044  files,  {'.mkv', '.webm', '.mp4'}
    ├── /workspace/llm_dataset/ActivityNet/videos : 3  files,  {'.zip'}
    '''
    indent += '    '
    c=0 
    
    ext_list=set()
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            print_tree(item_path, indent)
        else:
            c+=1
            ext_list.add(os.path.splitext(item_path)[1])
   
    if len(ext_list)!=0:
        print(indent+ '├── ' + directory, ':', c,' files, ', ext_list)
    else:
        pass

def main():
    if os.path.exists(dirPath) and os.path.isdir(dirPath):
        print_tree(dirPath)
    else:
        print("Incorrect path")

if __name__ == "__main__":
    main()
