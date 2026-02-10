**Design principle:** Prefer menus, context clicks, and visible UI over command-palette workflows.

## Git and history explorer

### Source Control 

* Purpose: Show current branch status and manage commits
* Location: Source Control icon on left sidebar

### Commit Graph (GitLens)

* Purpose: Show and interact with all git branches
* Location: Panel (bottom or right)
* How to open:
	* Open Source Control and click Show Commit Graph, or
	* Unhide the Panel (top-right toggle icon) and select GITLENS

### Compare a git commit with a local file (editable diff)

* Open the Commit Graph
* Right click a commit and select 'Compare working tree to here' 
* Expand 'XX files changed' and select a file from the list of files changed 
* Edit the local copy in the right pane. Changes are highlighted on the right margin. Hover over the middle gutter  on a changed line to get a revert arrow.  
* Close the Compare Tree when done to reduce clutter.

### Local history

* Purpose: Show uncommitted local file changes (independent of Git)
* Location: Explorer (file tree) -> Timeline (near the bottom if it's not expanded)
* Options: Filter Timeline to include/exclude Git commits and Local History entries. For commit diffs, use the Commit Graph workflow above.
* Click a Local History entry to bring up an editable diff against the current file.
* Use 'Restore content' to roll back the entire file, or hover over the middle gutter to restore individual blocks.

### Stash

* Purpose: Temporarily set aside local changes without committing them.
* PyCharm: VCS -> Shelve Changes (named shelves, per-file restore).
* VS Code equivalent: Git Stash.
* Workflow:
	* Source Control -> '...' menu -> Stash
	* Later: Source Control -> Stashes -> Apply or Pop
* Notes: Stashes are repository-wide. A stash saves all current uncommitted changes in the repository and can be applied on any branch. 
GitLens provides a Stashes view for browsing and applying them.

### Shelve emulator

* Purpose: Emulate PyCharm-style shelves using named, file-scoped change sets.
* Idea: Treat a shelf as a saved patch file rather than a Git object.
* Workflow:
	* Make local edits.
	* In Source Control, select the files to shelve.
	* Right click -> Export Changes...
	* Save the patch file in a folder such as `.vscode/shelves/` with a descriptive name.
* Restore:
	* Open the `.patch` file and apply it, or
	* Use Git tooling to apply the patch when ready.
* Notes:
	* Patch files are branch-independent and repository-independent.
	* Multiple shelves can coexist without interfering with Git history.
	* This most closely matches PyCharm’s named, per-file Shelve behavior.


## Utilities


### Outline (code structure)

* Purpose: Show code structure
* Location: Explorer (file tree) -> Outline 

### Compare two files

* Select a file.  Right click and choose 'Select for compare'.
* Select another file.  Right click and choose 'Compare with selected.'
* The first file will be on the left, the second on the right. 
* Hover over the middle gutter for copy-block arrows.  


## Run/Debug

* Goal: Support both one-click Run (no debugger) and predictable Debug (with stepping into project code).

* Model:
	* Editor top-right ▶︎ button = Run (no debugger).
	* Run & Debug panel ▶︎ button = Debug (uses launch.json, breakpoints, stepping).

* One-time setup for a new repo:
	* Open the repo root in VS Code.
	* Select the Python interpreter (conda env): 
      * Bottom right display when a .py file is active or
      * Command Palette -> Python: Select Interpreter.
	* Create `.vscode/launch.json` via Run & Debug -> create a launch.json -> Python.
	* Add debug configs (copy mbirjax/.vscode/launch.json into repo/.vscode/launch.json).

* Make `justMyCode` work for editable installs (already in launch.json for macOS):
	* If the package source lives inside the repo (e.g., repo_root/package_name):
		* Add the repo root to PYTHONPATH in the debug config:
			* macOS/Linux:
				* `PYTHONPATH=${workspaceFolder}:${env:PYTHONPATH}`
			* Windows:
				* `PYTHONPATH=${workspaceFolder};${env:PYTHONPATH}`
	* This allows Step Into to enter project code even when installed with `pip install -e .`.

* Keep Run and Debug behavior consistent:
	* Create a `.env` file at the repo root with:
		* `PYTHONPATH=.` 
	* Ensure Python › Env File points to `.env`.
	* Editor ▶︎ runs and Run & Debug ▶︎ debug now resolve imports the same way.

