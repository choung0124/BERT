import argparse
from extract_entities import extract_entities
from extract_relationships import extract_relationships

# Set up command-line arguments
parser = argparse.ArgumentParser(description='Extract entities and relationships from input text')
parser.add_argument('input_text', type=str, help='the input text to analyze')

# Parse command-line arguments
args = parser.parse_args()

# Extract entities and relationships from the input text
entities = extract_entities(args.input_text)
relationships = extract_relationships(args.input_text, entities)

# Print the extracted entities and relationships
print("Entities:")
for entity in entities:
    print(entity)

print("Relationships:")
for relationship in relationships:
    print(relationship)
