import re

# Define the log file path
log_file_path = "outputs/s3rec_-ml-1m-4309472.err"

# Open and read the log file
with open(log_file_path, "r") as file:
    lines = file.readlines()

# List to store extracted logs
extracted_logs = []
capture = False  # Flag to track when to start capturing relevant lines

# Iterate through each line in the log file
for line in lines:
    if "INFO  epoch" in line:  # Start capturing when epoch evaluation begins
        capture = True
        extracted_logs.append(line.strip())
    elif capture and "INFO  valid result" in line:  # Capture valid result header
        extracted_logs.append(line.strip())
    elif capture and re.search(r"(recall@|ndcg@|hit@)\d+", line):  # Capture metric lines
        extracted_logs.append(line.strip())
    elif capture and not line.strip():  # Stop capturing when we hit an empty line
        capture = False

# Print or save the extracted log
extracted_text = "\n".join(extracted_logs)
print(extracted_text)

# Optionally, save to a new file
with open("extracted_results.txt", "w") as output_file:
    output_file.write(extracted_text)
