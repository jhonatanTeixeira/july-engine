import json
import os

db_path = 'storage/db/tinydb.json'
if os.path.exists(db_path):
    with open(db_path, 'r') as f:
        db = json.load(f)
    
    settings = db.get('settings', {})
    for k, v in settings.items():
        if v.get('key') in ['WEB_SEARCH', 'REPOSITORY_SEARCH']:
            v['value']['backend'] = 'api'
    
    with open(db_path, 'w') as f:
        json.dump(db, f, indent=2)
    print("Database updated successfully.")
else:
    print("Database not found.")
