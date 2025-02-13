set -e

# Compile the metal shader to .air
xcrun -sdk macosx metal -c shader.metal -o shader.air

# Convert the .air file to .metallib
xcrun -sdk macosx metallib shader.air -o shader.metallib

echo "Shader compiled successfully to shader.metallib"