#!/usr/bin/env python3
import os
import glob

def check_env_files():
    print("🔍 환경 파일 확인 중...")
    env_files = glob.glob('.env*')
    
    if not env_files:
        print("  ℹ️ .env 파일이 없습니다")
        return
    
    for file in env_files:
        print(f"\n📄 {file}:")
        if file == '.env':
            print("  ⚠️  이 파일은 GitHub에 올라가면 안됩니다!")
        elif file == '.env.template':
            print("  ✅ 템플릿 파일 - GitHub 업로드 OK")
        elif file == '.env.backup':
            print("  ✅ 백업 파일 - 로컬에서만 보관")

def check_gitignore():
    print("\n🔍 .gitignore 확인 중...")
    
    if not os.path.exists('.gitignore'):
        print("❌ .gitignore 파일이 없습니다!")
        return False
    
    with open('.gitignore', 'r') as f:
        content = f.read()
    
    required_items = ['.env', '__pycache__/', 'logs/', 'data_cache/']
    all_protected = True
    
    for item in required_items:
        if item in content:
            print(f"  ✅ {item} - 보호됨")
        else:
            print(f"  ❌ {item} - 누락!")
            all_protected = False
    
    return all_protected

def main():
    print("🔒 보안 상태 확인")
    print("=" * 50)
    
    check_env_files()
    gitignore_ok = check_gitignore()
    
    print("\n📋 결과:")
    if gitignore_ok:
        print("✅ 보안 설정이 올바르게 되어 있습니다!")
    else:
        print("❌ 보안 설정에 문제가 있습니다.")

if __name__ == "__main__":
    main()
