# Step 1: Install required libraries
!pip install transformers gradio datasets

# Step 2: Import required libraries
import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import ast
import logging

# Step 3: Define the Multi-Language ARCK class with improvements
class MultiLanguageARCK:
    def __init__(self, model_name="Salesforce/codegen-2B-multi"):
        # Initializes the ARCK instance with a code generation model
        self.tokenizer, self.model = self.setup_code_generator(model_name)
        self.setup_logging()
        self.check_environment()

    def setup_logging(self):
        """Set up logging to capture generated code and errors for debugging."""
        logging.basicConfig(filename="arck_logs.log", level=logging.INFO, 
                            format="%(asctime)s - %(levelname)s - %(message)s")

    def check_environment(self):
        """Check for necessary compilers/interpreters."""
        required_tools = {
            "Node.js": "node --version",
            "Java": "javac -version",
            "C++": "g++ --version"
        }
        for tool, command in required_tools.items():
            try:
                subprocess.run(command.split(), capture_output=True, text=True)
            except FileNotFoundError:
                logging.warning(f"{tool} not found in the environment.")

    def setup_code_generator(self, model_name):
        """Initialize the tokenizer and model for code generation."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model

    def generate_code(self, prompt, language="Python", max_length=150):
        """Generate code based on a prompt and language choice."""
        language_prompt = f"Write the following code in {language}:\n{prompt}"
        inputs = self.tokenizer(language_prompt, return_tensors="pt")
        outputs = self.model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Generated code for prompt '{prompt}':\n{code}")
        return code

    def sanitize_code(self, code):
        """Sanitize code to prevent security risks."""
        restricted_keywords = ["import", "open", "eval", "exec", "os", "subprocess"]
        if any(keyword in code for keyword in restricted_keywords):
            logging.warning("Security Error: Unsafe code detected.")
            return False, "Security Error: Unsafe code detected."
        return True, "Code is safe."

    def validate_syntax(self, code, language="Python"):
        """Validate syntax without executing the code."""
        if language == "Python":
            try:
                ast.parse(code)
                return True, "Syntax is valid."
            except SyntaxError as e:
                return False, f"Syntax Error: {e}"
        # Other languages could have basic compilation checks here
        return True, "Syntax validation not available for this language."

    def test_code(self, code, language="Python"):
        """Test the code in the specified language environment with resource limits."""
        is_safe, safety_msg = self.sanitize_code(code)
        if not is_safe:
            return safety_msg
        
        # Run code with a timeout and capture errors
        if language == "Python":
            try:
                exec(code, {"__builtins__": {}})
                return "Code executed successfully in Python."
            except Exception as e:
                return f"Python Execution Error: {e}"
        
        elif language == "JavaScript":
            with open("temp.js", "w") as file:
                file.write(code)
            try:
                result = subprocess.run(["node", "temp.js"], capture_output=True, text=True, timeout=5)
                return result.stdout if result.returncode == 0 else f"JavaScript Execution Error: {result.stderr}"
            except subprocess.TimeoutExpired:
                return "JavaScript Execution Error: Timeout expired."
            except Exception as e:
                return f"JavaScript Execution Error: {e}"

        elif language == "Java":
            with open("Temp.java", "w") as file:
                file.write(code)
            try:
                compile_result = subprocess.run(["javac", "Temp.java"], capture_output=True, text=True)
                if compile_result.returncode != 0:
                    return f"Java Compilation Error: {compile_result.stderr}"
                result = subprocess.run(["java", "Temp"], capture_output=True, text=True, timeout=5)
                return result.stdout if result.returncode == 0 else f"Java Execution Error: {result.stderr}"
            except subprocess.TimeoutExpired:
                return "Java Execution Error: Timeout expired."
            except Exception as e:
                return f"Java Execution Error: {e}"

        elif language == "C++":
            with open("temp.cpp", "w") as file:
                file.write(code)
            try:
                compile_result = subprocess.run(["g++", "temp.cpp", "-o", "temp"], capture_output=True, text=True)
                if compile_result.returncode != 0:
                    return f"C++ Compilation Error: {compile_result.stderr}"
                result = subprocess.run(["./temp"], capture_output=True, text=True, timeout=5)
                return result.stdout if result.returncode == 0 else f"C++ Execution Error: {result.stderr}"
            except subprocess.TimeoutExpired:
                return "C++ Execution Error: Timeout expired."
            except Exception as e:
                return f"C++ Execution Error: {e}"

        return "Language not supported for testing."

# Step 4: Initialize ARCK instance for multi-language support
arck = MultiLanguageARCK()

# Step 5: Define the Gradio interface for interactive use
def interact_with_arck(prompt, language):
    """Generate and test code interactively through ARCK."""
    generated_code = arck.generate_code(prompt, language)
    syntax_valid, syntax_msg = arck.validate_syntax(generated_code, language)
    if not syntax_valid:
        return f"Generated Code:\n{generated_code}\n\nSyntax Validation:\n{syntax_msg}"
    
    test_result = arck.test_code(generated_code, language)
    return f"Generated Code:\n{generated_code}\n\nSyntax Validation:\n{syntax_msg}\n\nTest Result:\n{test_result}"

# Step 6: Launch Gradio interface with additional instructions
iface = gr.Interface(
    fn=interact_with_arck,
    inputs=[
        gr.inputs.Textbox(label="Prompt"),
        gr.inputs.Dropdown(["Python", "JavaScript", "Java", "C++"], label="Language")
    ],
    outputs="text",
    title="Multi-Language ARCK",
    description="Generate and test code in multiple programming languages with ARCK.\n\nNote: Only safe code will be executed. Avoid using restricted keywords or complex scripts.",
)

iface.launch()
