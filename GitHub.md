### 0. The Mental Model First
Before commands, understand the shape of what you're working with Code
Think of it as four rooms your code passes through:

Working Directory — the actual files on your disk, as you edit them.

Staging Area (the "index") — a holding pen. You choose which changes go into the next commit.

Local Repo — your commit history, stored in .git/, on your machine only.

Remote Repo — the copy on GitHub. Nothing leaves your machine until you push.
A branch is just a movable label pointing at one commit in that history. main is simply the name of the default branch — not special to Git itself.

When you create a new branch, you're just adding a new label at the same commit ,,As you commit on feature, it moves forward independently:
Everything below builds on this picture.

### 1. One-Time Setup

Check what's set: 
Bash
  `git config --global user.name "Your Name"`
  `git config --global user.email "you@example.com"` 

## 2. Pushing a File to GitHub (from scratch)

#### Step 1 — Go to the exact folder your file is in.

`Bash`
`cd "C:\path\to\your\folder"`

cd moves you there. Typing a path alone does not.

#### Step 2 — Turn the folder into a Git repo (only once per folder).

`git init`

#### Step 3 — Stage the file.

`git add filename.py`

git add . stages everything in the folder — be careful with this on folders like Desktop.

#### Step 4 — Commit (save a snapshot to local history).

`Bash`
`git commit -m "Add filename.py"`

#### Step 5 — Name your default branch main (modern convention).

`Bash`
`git branch -M main`

#### Step 6 — Connect to the GitHub repo.

`Bash`
`git remote add origin https://github.com/USERNAME/REPO.git`

`git remote -v` - to check if it is correctly set


Step 7 — Push.

Bash
`git push -u origin main`

The -u links your local main to remote main so future pushes are just git push.
If push is rejected ("fetch first" / "remote contains work you don't have"):

This means GitHub already has commits your local repo doesn't (e.g. a README made when the repo was created). 

Pull first:
Bash
`git pull origin main --allow-unrelated-histories`
`git push -u origin main`

### 3. Updating a File You Already Pushed
Once a repo is already connected, updating is just three commands, every time:

Bash
`git add filename.py`
`git commit -m "Describe what changed"`
`git push`

No init, no remote add — those are one-time setup steps.

Check status anytime to see what's changed but not yet staged/committed:
Bash
`git status`
See your commit history:
Bash
`git log --oneline`

### 4. Branches — Creating and Pushing to Them
Branches let you work on something (a new feature, an experiment) without touching main until it's ready.

Create a new branch and switch to it in one step:

Bash
`git checkout -b feature-name`

(Modern equivalent: git switch -c feature-name)

Add/edit your file, then commit as usual:

Bash
`git add newfile.py`
`git commit -m "Work on new feature"`

Push the branch to GitHub (creates it remotely too):

Bash
`git push -u origin feature-name`

> Switch between branches:
 
 Bash
`git checkout main`
`git checkout feature-name`

See all branches (local):

Bash
`git branch`

See all branches (including remote):

Bash
`git branch -a`

### 5. Merging a Branch into Main

Once your branch's work is ready, bring it into main:

Bash
`git checkout main     ->  # go to the branch you want to merge INTO`

`git pull origin main      # make sure local main is up to date`

`git merge feature-name     # merge feature-name into main`

`git push origin main       # push the updated main`

Picture it:
Code
Before merge:
main:      A---B---C
                 \
  feature:                            D---E

After merge:
main:      A---B---C---------F   (F = merge commit)
              \       /
feature:                D-----E


#### 6. Updating a File Inside a Branch (After It Already Exists)

Same three-step pattern as before, just make sure you're on that branch first:

Bash
`git checkout feature-name # edit your file`
`git add filename.py`
`git commit -m "Update file in feature branch"`
`git push origin feature-name`

### 7. Merging All Branches Into One Single Branch

If you have several branches and want to consolidate everything into one (say, main):

Option A — Merge them one at a time (safest, recommended):

Bash
`git checkout main`
`git merge branch-1`
`git merge branch-2`
`git merge branch-3`
`git push origin main`

Resolve conflicts between each merge if they arise, using Section 5's steps.

Option B — See what will merge before doing it:

Bash
`git log branch-name --oneline    # preview commits on that branch`
`git diff main branch-name         # see what's different`

After merging everything into main, clean up old branches:

Bash
`git branch -d branch-1              # delete local branch (only if merged)`
`git push origin --delete branch-1   # delete it on GitHub too`

Use -D (capital) instead of -d only if you want to force-delete a branch that isn't fully merged — use with caution, this can lose work.

### 8. Quick Reference Cheat-Sheet

Start tracking a folder
> git init

Stage a file
> git add filename

Stage everything
> git add .

Save a snapshot
> git commit -m "message"

Connect to GitHub
> git remote add origin URL

Check remote URL
> git remote -v

Change remote URL
> git remote set-url origin URL

Push first time
> git push -u origin main

Push after that
> git push

Pull remote changes
> git pull

Create + switch branch
> git checkout -b name

Switch branch
> git checkout name

List branches
> git branch (add -a for remote too)

Merge branch into current
> git merge name

Delete local branch
> git branch -d name

Delete remote branch
> git push origin --delete name

See history
> git log --oneline

See changed/staged files
> git status

See exact line changes
> git diff
