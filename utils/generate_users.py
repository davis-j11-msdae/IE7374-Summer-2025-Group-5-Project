#!/usr/bin/env python3
"""
Generate user accounts for the storytelling system.
Creates 5 users for each age group with standard naming, plus admin user.
"""

import os
import pandas as pd
from pathlib import Path

def generate_users():
    """Generate user accounts file with admin functionality."""
    users_dir = Path("data/users")
    users_dir.mkdir(parents=True, exist_ok=True)
    
    users_file = users_dir / "users.txt"
    
    # Load existing users if file exists
    existing_users = {}
    if users_file.exists():
        try:
            existing_df = pd.read_csv(users_file)
            for _, row in existing_df.iterrows():
                existing_users[row['username']] = {
                    'age': row['age'],
                    'password': row['password'],
                    'admin': row.get('admin', 0)
                }
            print(f"Loaded {len(existing_users)} existing users")
        except Exception as e:
            print(f"Error loading existing users: {e}")
    
    age_groups = {
        'child': [3, 4, 5, 5, 4],
        'kid': [7, 9, 11, 8, 10],
        'teen': [14, 16, 17, 15, 13],
        'adult': [25, 32, 28, 45, 38]
    }
    
    # Add standard users if they don't exist
    for group, ages in age_groups.items():
        for i, age in enumerate(ages, 1):
            username = f"{group}_{i}"
            if username not in existing_users:
                existing_users[username] = {
                    'age': age,
                    'password': 'test',
                    'admin': 0
                }
    
    # Add admin user if it doesn't exist
    if 'admin' not in existing_users:
        existing_users['admin'] = {
            'age': 25,
            'password': 'admin',
            'admin': 1
        }
    
    # Write all users to file
    with open(users_file, 'w') as f:
        f.write("username,age,password,admin\n")
        for username, data in existing_users.items():
            f.write(f"{username},{data['age']},{data['password']},{data['admin']}\n")
    
    total_users = len(existing_users)
    admin_count = sum(1 for data in existing_users.values() if data['admin'] == 1)
    regular_count = total_users - admin_count
    
    print(f"Generated users file: {users_file}")
    print(f"Total users: {total_users} ({admin_count} admin, {regular_count} regular)")

if __name__ == "__main__":
    generate_users()