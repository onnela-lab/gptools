import argparse
import os
import re
import subprocess


def replace(text: str, lookup):
    if isinstance(text, bytes):
        lookup = {old.encode(): new.encode() for old, new in lookup.items()}
    for old, new in lookup.items():
        text = text.replace(old, new)
    return text


parser = argparse.ArgumentParser()
parser.add_argument("template_name", help="name of the template")
parser.add_argument("--template_user", "-u", help="github username")
parser.add_argument("--template_repo", "-r", help="github repository name")
parser.add_argument("--keep_script", help="don\"t delete this script", action="store_true")
args = parser.parse_args()

lookup = {
    "TEMPLATE_NAME": args.template_name,
}
# Try to determine the github username and repository if not given.
if args.template_user and args.template_repo:
    lookup["TEMPLATE_USER"] = args.template_user
    lookup["TEMPLATE_REPO"] = args.template_repo
else:
    output = subprocess.check_output(["git", "remote", "-v"]).decode()
    match = re.search(r"git@github.com:(?P<user>[\w-]+)/(?P<repo>[\w-]+).git", output)
    if not match:
        raise ValueError("cannot determine the github user or repository name")
    lookup["TEMPLATE_USER"] = args.template_user or match.group("user")
    lookup["TEMPLATE_REPO"] = args.template_repo or match.group("repo")

# Apply all the replacements by walking the directory tree.
for path, dirnames, filenames in os.walk("."):
    # Rename directories and files if required.
    for names in [dirnames, filenames]:
        for i, name in enumerate(names):
            if (newname := replace(name, lookup)) != name:
                oldpath = os.path.join(path, name)
                newpath = os.path.join(path, newname)
                os.rename(oldpath, newpath)
                print(f"renamed {oldpath} -> {newpath}")
                names[i] = newname

    # Don"t recurse into the git directory.
    try:
        dirnames.remove(".git")
    except ValueError:
        pass

    # Replace the file content.
    for filename in filenames:
        # Don"t mess with this file.
        if __file__.endswith(filename):
            continue
        filename = os.path.join(path, filename)
        with open(filename, "rb") as fp:
            text = fp.read()

        if (newtext := replace(text, lookup)) != text:
            with open(filename, "wb") as fp:
                fp.write(newtext)
            print(f"replaced content in {filename}")

# Delete this script.
if not args.keep_script:
    os.unlink(__file__)
    print(f"removed {__file__}")
