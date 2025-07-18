#!/usr/bin/env python3
"""
Generate user accounts for the storytelling system.
Creates 5 users for each age group with standard naming.
"""

import os
from pathlib import Path


def generate_users():
    """Generate user accounts file."""
    users_dir = Path("data/users")
    users_dir.mkdir(parents=True, exist_ok=True)

    users_file = users_dir / "users.txt"

    age_groups = {
        'child': [3, 4, 5, 5, 4],
        'kid': [7, 9, 11, 8, 10],
        'teen': [14, 16, 17, 15, 13],
        'adult': [25, 32, 28, 45, 38]
    }

    with open(users_file, 'w') as f:
        f.write("username,age,password\n")

        for group, ages in age_groups.items():
            for i, age in enumerate(ages, 1):
                username = f"{group}_{i}"
                f.write(f"{username},{age},test\n")

    print(f"Generated users file: {users_file}")
    print(f"Created 20 users (5 per age group)")


if __name__ == "__main__":
    generate_users()