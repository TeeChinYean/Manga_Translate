#!/bin/bash
set -e

echo "========================================="
echo "   Antigravity C# Engine Setup Script    "
echo "========================================="

echo "[1/4] Checking and installing .NET 8 SDK..."
if ! command -v dotnet &> /dev/null; then
    echo "dotnet command not found. Installing dotnet-sdk-8.0..."
    sudo apt update
    sudo apt install -y dotnet-sdk-8.0
else
    # Check if the SDK is installed, not just the host
    if ! dotnet --list-sdks | grep -q "8.0"; then
        echo "dotnet SDK 8.0 not found. Installing..."
        sudo apt update
        sudo apt install -y dotnet-sdk-8.0
    else
        echo "✅ .NET 8 SDK is already installed."
    fi
fi

echo "[2/4] Navigating to C# Project Directory..."
cd /home/tcy/pdf_translate/PdfTranslate.CSharp

echo "[3/4] Restoring NuGet Packages (pythonnet)..."
dotnet restore

echo "[4/4] Building and Launching the C# Engine..."
echo "Starting Antigravity C# Orchestrator..."
dotnet run
