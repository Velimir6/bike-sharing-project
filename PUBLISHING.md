# Publish the existing project history

The local repository is on branch `main`. Keep its `.git` directory: it contains the substantive development history required by the assignment. Uploading the files as one new commit would discard that history.

Create an empty repository in the intended GitHub account. Do not initialize it with an unrelated README, license or `.gitignore`. Then, from this project folder, add its actual URL and push the existing branch:

```text
git remote add origin YOUR_ACTUAL_REPOSITORY_URL
git push -u origin main
```

The placeholder must be replaced with the real repository URL. Authenticate using the normal GitHub sign-in flow; do not put access tokens into project files. If a remote already exists, inspect it with `git remote -v` before changing anything.

Verify that the GitHub page shows the notebook with its outputs, source data, documentation and more than ten commits. The account owner must confirm examiner access and submit the repository link through the course's prescribed channel before its deadline. This document does not indicate that publication or submission has already occurred.

Agent-authored commits use `Codex <codex@localhost>`. See `ASSISTANCE.md` for the contribution statement. New human-authored commits can use the user's chosen Git name and email; no global Git settings were changed during project preparation.
