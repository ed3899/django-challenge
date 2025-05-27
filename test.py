import os

for (root,dirs,files) in os.walk('/workspaces/django-challenge/docs-sm', topdown=True):
    print (f"root: {root}")
    print (f"dirs: {dirs}")
    print (files)
    print ('--------------------------------')