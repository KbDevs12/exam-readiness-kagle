import sys
import importlib.util

packages = [
    "streamlit",
    "streamlit_option_menu",
    "numpy",
    "pandas",
    "matplotlib",
    "kagglehub",
]

'''Untuk memeriksa apakah semua package yang dibutuhkan sudah terinstall di Python yang sedang dipakai.'''

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print()

missing = []
for package in packages:
    found = importlib.util.find_spec(package) is not None
    print(f"{package:24}:", "OK" if found else "NOT FOUND")
    if not found:
        missing.append(package)

if missing:
    print("\nAda package yang belum terinstall di Python ini.")
    print("Jalankan:")
    print(f'"{sys.executable}" -m pip install -r requirements.txt')
else:
    print("\nSemua package terdeteksi di Python yang sedang dipakai.")
