#!/usr/bin/env python3
"""
Script para instalar dependencias del backend ExoVerse
Maneja problemas comunes de instalación en Windows
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Ejecutar comando y manejar errores"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}:")
        print(f"   Comando: {command}")
        print(f"   Error: {e.stderr}")
        return False

def install_dependencies():
    """Instalar dependencias paso a paso"""
    print("🚀 Instalando dependencias de ExoVerse Backend...")
    print("=" * 50)
    
    # Actualizar pip primero
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Actualizando pip"):
        print("⚠️  Continuando sin actualizar pip...")
    
    # Instalar setuptools primero
    if not run_command(f"{sys.executable} -m pip install --upgrade setuptools wheel", "Instalando setuptools y wheel"):
        print("❌ Error crítico: No se puede instalar setuptools")
        return False
    
    # Instalar dependencias básicas primero
    basic_deps = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.5.0",
        "python-multipart==0.0.6"
    ]
    
    print("\n📦 Instalando dependencias básicas...")
    for dep in basic_deps:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Instalando {dep}"):
            print(f"⚠️  Fallo al instalar {dep}, continuando...")
    
    # Instalar numpy
    if not run_command(f"{sys.executable} -m pip install numpy==1.24.3", "Instalando numpy"):
        print("⚠️  Fallo al instalar numpy específico, probando versión más reciente...")
        run_command(f"{sys.executable} -m pip install numpy", "Instalando numpy (última versión)")
    
    # Instalar scikit-learn
    if not run_command(f"{sys.executable} -m pip install scikit-learn==1.3.0", "Instalando scikit-learn"):
        print("⚠️  Fallo al instalar scikit-learn específico, probando versión más reciente...")
        run_command(f"{sys.executable} -m pip install scikit-learn", "Instalando scikit-learn (última versión)")
    
    # Instalar pandas
    if not run_command(f"{sys.executable} -m pip install pandas==2.0.3", "Instalando pandas"):
        print("⚠️  Fallo al instalar pandas específico, probando versión más reciente...")
        run_command(f"{sys.executable} -m pip install pandas", "Instalando pandas (última versión)")
    
    # Instalar dependencias opcionales
    optional_deps = [
        "joblib==1.3.2",
        "matplotlib==3.7.2",
        "seaborn==0.12.2"
    ]
    
    print("\n📦 Instalando dependencias opcionales...")
    for dep in optional_deps:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Instalando {dep}"):
            print(f"⚠️  Fallo al instalar {dep}, continuando sin esta dependencia...")
    
    print("\n✅ Instalación completada!")
    return True

def test_installation():
    """Probar que las dependencias principales funcionan"""
    print("\n🧪 Probando instalación...")
    
    try:
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__}")
    except ImportError:
        print("❌ FastAPI no disponible")
        return False
    
    try:
        import uvicorn
        print(f"✅ Uvicorn {uvicorn.__version__}")
    except ImportError:
        print("❌ Uvicorn no disponible")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy {numpy.__version__}")
    except ImportError:
        print("❌ NumPy no disponible")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn no disponible")
        return False
    
    try:
        import pandas
        print(f"✅ Pandas {pandas.__version__}")
    except ImportError:
        print("❌ Pandas no disponible")
        return False
    
    print("\n🎉 ¡Todas las dependencias principales están funcionando!")
    return True

def main():
    """Función principal"""
    print("ExoVerse Backend - Instalador de Dependencias")
    print("=" * 50)
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("main.py"):
        print("❌ Error: No se encontró main.py")
        print("   Asegúrate de ejecutar este script desde el directorio backend/")
        return False
    
    # Instalar dependencias
    if not install_dependencies():
        print("\n❌ Error en la instalación de dependencias")
        return False
    
    # Probar instalación
    if not test_installation():
        print("\n⚠️  Algunas dependencias no están funcionando correctamente")
        print("   Pero puedes intentar ejecutar el servidor de todas formas")
    
    print("\n🚀 ¡Listo! Ahora puedes ejecutar:")
    print("   python run_server.py")
    print("   o")
    print("   python main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


