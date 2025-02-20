"""

python main_v2.py --topic "Build your own lightsaber" --output-dir ./results --do-research --do-generate-outline --do-generate-article


Result saved in Results > report_1 > report_1.md

"""

import os
import argparse
import json
from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.lm import ClaudeModel
from knowledge_storm.rm import SerperRM
import shutil
import replicate  # Import the replicate library
import random
import re
import time
import backoff


# Function to introduce grammatical errors
def introduce_grammatical_errors(text, error_rate=0.05):
    """
    Introduce subtle grammatical errors in the text to make it seem more human-written.
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    modified_sentences = []

    for sentence in sentences:
        if random.random() < error_rate:
            words = sentence.split()
            if len(words) > 2:
                # Introduce a subject-verb agreement error
                if random.random() < 0.5:
                    if words[0].lower() not in ['i', 'we', 'you', 'they'] and not words[0].endswith('s'):
                        if words[1].lower() == 'is':
                            words[1] = 'are'
                        elif words[1].lower() == 'was':
                            words[1] = 'were'
                    sentence = ' '.join(words)
                # Introduce a comma splice
                else:
                    if len(words) > 5:
                        split_index = random.randint(1, len(words) - 2)
                        words.insert(split_index, ',')
                        sentence = ' '.join(words)
        modified_sentences.append(sentence)

    return ' '.join(modified_sentences)

def generate_image(prompt, image_path):
    """
    Generates an image using Replicate based on the given prompt.
    """
    try:
        # Call the Replicate API to generate the image
        image_url = replicate.run(
            "google/imagen-3-fast",
            input={
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "safety_filter_level": "block_medium_and_above"
            }
        )

        # Download the image and save it
        if image_url:  # Check if image URL is available
            print(f"Image generated successfully. URL: {image_url}")
        else:
            print("Image generation failed.")
            return None
        return image_url

    except Exception as e:
        print(f"Error generating image: {e}")
        return None

@backoff.on_exception(backoff.expo, Exception, max_tries=2)
def run_with_retry(runner, topic_with_keywords):
    """
    Run the STORM process with retry mechanism and a 3-second wait after each attempt.
    """
    try:
        # Run the STORM process
        runner.run(
            topic=topic_with_keywords,
            do_research=args.do_research,
            do_generate_outline=args.do_generate_outline,
            do_generate_article=args.do_generate_article,
            do_polish_article=False,
        )
        runner.post_run()
    except Exception as e:
        print(f"Error during STORM process: {e}. Waiting 3 seconds before retrying...")
        time.sleep(3)  # Add a 3-second wait before retrying
        raise  # Re-raise the exception to trigger the backoff mechanism


def main(args):
    # Hardcoded API keys
    os.environ[
        "ANTHROPIC_API_KEY"] = ""
    os.environ["SERPER_API_KEY"] = ""
    os.environ["REPLICATE_API_TOKEN"] = "" # SET THE API KEY
    # Fixed directory name
    fixed_dir_name = "report_1"
    topic_dir = os.path.join(args.output_dir, fixed_dir_name)

    # Ensure the fixed directory exists
    os.makedirs(topic_dir, exist_ok=True)

    # Initialize language model configurations
    lm_configs = STORMWikiLMConfigs()
    claude_kwargs = {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "temperature": 1.0,
        "top_p": 0.9,
    }

    conv_simulator_lm = ClaudeModel(
        model="claude-3-haiku-20240307", max_tokens=2096, **claude_kwargs
    )
    question_asker_lm = ClaudeModel(
        model="claude-3-sonnet-20240229", max_tokens=2096, **claude_kwargs
    )
    outline_gen_lm = ClaudeModel(
        model="claude-3-opus-20240229", max_tokens=2096, **claude_kwargs
    )
    article_gen_lm = ClaudeModel(
        model="claude-3-opus-20240229", max_tokens=2096, **claude_kwargs
    )

    lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    lm_configs.set_question_asker_lm(question_asker_lm)
    lm_configs.set_outline_gen_lm(outline_gen_lm)
    lm_configs.set_article_gen_lm(article_gen_lm)

    # Initialize retrieval module
    engine_args = STORMWikiRunnerArguments(
        output_dir=topic_dir,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
    )
    rm = SerperRM(serper_search_api_key=os.getenv("SERPER_API_KEY"),
                  query_params={"autocorrect": True, "num": 10, "page": 1})

    # Initialize STORMWikiRunner
    runner = STORMWikiRunner(engine_args, lm_configs, rm)

    # Combine topic and keywords for the article generation
    topic_with_keywords = f"{args.topic} {args.keywords}"

    try:
        run_with_retry(runner, topic_with_keywords)
    except Exception as e:
        print(f"Final Error: {e}")

    # Get the topic-based directory name
    sanitized_topic = topic_with_keywords.replace(" ", "_")
    generated_dir = os.path.join(topic_dir, sanitized_topic)

    # Move files from topic-based directory to report_1
    if os.path.exists(generated_dir):
        for file_name in os.listdir(generated_dir):
            src_path = os.path.join(generated_dir, file_name)
            dst_path = os.path.join(topic_dir, file_name)
            shutil.move(src_path, dst_path)
        # Remove the empty topic-based directory
        os.rmdir(generated_dir)

    # Debugging: Check if the directory and files exist
    print(f"Checking directory: {topic_dir}")
    if os.path.exists(topic_dir):
        print(f"Directory exists: {topic_dir}")
        print(f"Contents of directory: {os.listdir(topic_dir)}")
    else:
        print(f"Directory does not exist: {topic_dir}")

    # Retrieve the generated article and references from the output directory
    article_file_path = os.path.join(topic_dir, "storm_gen_article.txt")
    references_file_path = os.path.join(topic_dir, "url_to_info.json")

    if not os.path.exists(article_file_path):
        raise FileNotFoundError(f"The article file does not exist: {article_file_path}")

    if not os.path.exists(references_file_path):
        raise FileNotFoundError(f"The references file does not exist: {references_file_path}")

    with open(article_file_path, 'r') as article_file:
        generated_article = article_file.read()

    with open(references_file_path, 'r') as references_file:
        references = references_file.read()

    # Sanitize article
    humanized_article = introduce_grammatical_errors(generated_article)

    # # Define a location for the image.
    # output_image_path = os.path.join(args.output_dir, f"{fixed_dir_name}_lightsaber_image.jpg")

    # Save the generated article and references to a Markdown file
    output_file_path = os.path.join(args.output_dir, f"{fixed_dir_name}.md")
    with open(output_file_path, 'w') as md_file:
        md_file.write(f"# {args.topic}\n\n")

        # Generate and embed the image into the Markdown file
        image_prompt = f"{args.topic} {args.keywords}"
        image_url = generate_image(image_prompt, output_file_path)
        if image_url:
            md_file.write(f"<img src='{image_url}' alt='Generated Image' width='500'/>\n\n")
        else:
            md_file.write("Image generation failed. A placeholder will show until a new image can be generated.\n\n")
        md_file.write(humanized_article)
        md_file.write("\n\n## References\n\n")
        md_file.write(references)

    print(f"Generated article and references saved to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the STORM Wiki pipeline on a given topic.")
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="The topic to generate the article for.",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="",
        help="Additional keywords to include in the article generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/serper",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--max-thread-num",
        type=str,
        default=3,
        help="Maximum number of threads to use.",
    )
    parser.add_argument(
        "--do-research",
        action="store_true",
        help="If True, simulate conversation to research the topic.",
    )
    parser.add_argument(
        "--do-generate-outline",
        action="store_true",
        help="If True, generate an outline for the topic.",
    )
    parser.add_argument(
        "--do-generate-article",
        action="store_true",
        help="If True, generate an article for the topic.",
    )
    parser.add_argument(
        "--max-conv-turn",
        type=int,
        default=5,
        help="Maximum number of questions in conversational question asking.",
    )
    parser.add_argument(
        "--max-perspective",
        type=int,
        default=3,
        help="Maximum number of perspectives to consider.",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=10,
        help="Top k search results to consider for each search query.",
    )

    args = parser.parse_args()
    main(args)


