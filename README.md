# Instructions

## Package installation

- Python 3.9 or higher
- Install latest pip packages in the normal case, i.e remove version numbers.
- If it doesn't work, install the exact packages as in the requirements.txt using
```
pip install -r requirements.txt
```

## Usage

- Notebooks can be run directly and modifications can be made in the notebook itself to produce desired results.
- `graph_tools.py` houses all the modules created for various purposes and in the core part of the project.
- If running in linux, make the showg executable by `chmod u+x showg` and change line 12 in `graph_tools.py` as 
```
    output = subprocess.check_output(["./showg", "-A", f_path], universal_newlines=True)
```
