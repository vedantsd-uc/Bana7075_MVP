
from pathlib import Path
from datapipeline.bike_data_pipeline import load_data, validate_data


# ------------------------------------------------------
# Location of data quality test files
# ------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "data_quality"


# ------------------------------------------------------
# Test cases: (filename, is_train)
# ------------------------------------------------------
TEST_CASES = [
    ("missing_column.csv", True),
    ("negative_count.csv", True),
    ("invalid_datetime.csv", True),
]


def run_data_quality_tests():
    print("Starting Data Quality Tests...\n")

    for filename, is_train in TEST_CASES:
        file_path = TEST_DATA_DIR / filename
        print(f"Testing file: {filename}")

        try:
            df = load_data(file_path)
            validate_data(df, is_train=is_train)

            # If we reach here, the validation DID NOT fail (unexpected)
            print("FAILED — Expected validation error, but none occurred\n")

        except Exception as e:
            # This is the expected behavior
            print(f"PASSED — Caught expected error:")
            print(f"   {e}\n")


if __name__ == "__main__":
    run_data_quality_tests()
