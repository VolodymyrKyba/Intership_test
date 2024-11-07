from transformers import pipeline

# Loading the saved model and tokenizer
ner_pipeline = pipeline("ner", model="./best_model", tokenizer="./saved_mode2")

# Text for inference
text = "One of the most breathtaking places for travelers is the Himalayas, home to Mount Khushkak and Kanchenjunga, which attract climbers from all over the world. In Europe, Mont Blanc, the highest peak in the Alps, is also a popular destination for those seeking to conquer summits. In South America, the Andes mountain range stretches for thousands of kilometers, including famous peaks such as Aconcagua and Huascarán.In Africa, Mount Kilimanjaro stands as the highest peak on the continent, easily accessible without specialized climbing gear. Another impressive peak is Mount Meru, located in Tanzania, which is known for its rugged terrain. Moving to North America, the Rocky Mountains boast some of the most iconic mountain landscapes, with peaks like Mount Elbert and Mount Whitney standing tall.In the Pacific region, Mount Fuji in Japan is not only a cultural symbol but also a popular climbing destination. The mountains of New Zealand, like Mount Cook and Mount Taranaki, offer incredible hiking and climbing experiences. Finally, the Alps and the Carpathians continue to draw mountaineers and adventurers looking for challenging terrain and magnificent views."

results = ner_pipeline(text)

# Function to merge subtokens into full mountain names
def extract_full_entities(results):
    merged_entities = []
    current_entity = {
        "entity": None,
        "score": 0,
        "word": "",
        "start": None,
        "end": None,
    }

    for entity in results:
        if entity["word"].startswith("##"):
            current_entity["word"] += entity["word"].replace("##", "")
            current_entity["end"] = entity["end"]
            current_entity["score"] = min(current_entity["score"], entity["score"])
        else:
            if current_entity["entity"] is not None:
                merged_entities.append(current_entity)
            current_entity = {
                "entity": entity["entity"],
                "score": entity["score"],
                "word": entity["word"],
                "start": entity["start"],
                "end": entity["end"],
            }

    if current_entity["entity"] is not None:
        merged_entities.append(current_entity)

    return merged_entities


# Get the result with merged mountain names
final_entities = extract_full_entities(results)
final_list = []
# Output merged mountain names
for entity in final_entities:
    if entity["entity"] == "LABEL_1" or entity["entity"] == "LABEL_2":
        final_list.append(entity["word"].capitalize())
print(final_list)
