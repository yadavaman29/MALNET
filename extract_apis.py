import pefile
import sys

def extract_apis(filepath):
    print(f"Extracting APIs from: {filepath}...\n")
    try:
        pe = pefile.PE(filepath)
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for lib in pe.DIRECTORY_ENTRY_IMPORT:
                print(f"[{lib.dll.decode('utf-8', errors='ignore')}]")
                for imp in lib.imports:
                    if imp.name:
                        print(f"  - {imp.name.decode('utf-8', errors='ignore')}")
        else:
            print("No imports found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else r"dataset\header_compiler.exe"
    extract_apis(filepath)
