# This file is for site maintenance operations
# To be runned on linux-based server only

# Clean the site files
rm -r DOC/site

# Get the latest version of the site
git pull

# Get in the site directory
cd DOC

# Build the site - and nginx / apache will serve it
mkdocs build

# Deploy the site
# mkdocs serve