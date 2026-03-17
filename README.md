# openpi 

### train your own model

modify `config.py`

modify `policies/maniskill_policy.py`

read `run.sh` and `resume_run.sh`

### Push this project to your own GitHub

**If you already cloned upstream and want your fork as `origin`:**

```bash
cd openpi
git remote rename origin upstream    # optional: keep upstream for pulling updates
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git add -A
git status   # confirm no checkpoints / ckpt / .venv listed
git commit -m "Fork: openpi + local training scripts"
git branch -M main
git push -u origin main
```

**If this folder is not a git repo yet:**

```bash
cd openpi
git init
git add -A
git commit -m "Initial commit: openpi fork"
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u origin main
```

Create an **empty** repository on GitHub first (no README/license) so the first push succeeds. Use SSH URL `git@github.com:USER/REPO.git` if you prefer.

### Pull updates from upstream later

```bash
git fetch upstream
git merge upstream/main    # or rebase
```

---


