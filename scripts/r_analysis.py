import subprocess

def main():
    result = subprocess.run(["Rscript", "scripts/intersectionalBiasPlot.R"], check=True)
    print("R script completed", result.returncode)

if __name__ == "__main__":
    main()
