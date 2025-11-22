import pandas as pd
import re

raw_file = "lab_ranges.csv"
clean_file = "lab_ranges_clean.csv"

df = pd.read_csv(raw_file, header=2)  # skip first two rows

clean_rows = []

for index, row in df.iterrows():
    test_name = str(row[0]).strip()
    ranges = str(row[1]).strip()

    # Skip "See Vitamin D metabolites"
    if ranges.lower().startswith("see"):
        continue

    # Split gender/condition segments
    segments = re.split(r";|\n", ranges)

    for segment in segments:
        segment = segment.strip()

        # Pattern for ranges like: Female: 6.8–29.3 μg/mL
        match = re.search(
            r"(?:(Female|Male|follicular|luteal|postmenopausal|adult)[\s,:]*)?"
            r"(<)?\s*([\d\.]+)\s*[–\-]?\s*([\d\.]+)?\s*([a-zA-Z/μ%]+)?",
            segment,
            flags=re.IGNORECASE
        )

        if match:
            group = match.group(1) or ""
            is_less_than = match.group(2)
            min_val = match.group(3)
            max_val = match.group(4) or min_val  # if only one bound
            unit = match.group(5) or ""

            # Convert numbers
            min_val = float(min_val)
            max_val = float(max_val)

            # Handle "< value" case
            if is_less_than:
                min_val = 0.0

            clean_rows.append([test_name, group, min_val, max_val, unit])

# Save clean structured format
clean_df = pd.DataFrame(clean_rows, columns=["Test Name", "Group", "Normal_Min", "Normal_Max", "Unit"])
clean_df.to_csv(clean_file, index=False)

print("✅ Clean structured CSV created:", clean_file)
