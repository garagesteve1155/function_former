"""
Prompt Improvements

Variable fixes

Hardcoded indentation fix (Will need more testing)

Improved output logging from running scripts

Improved data shown on Chat-Coding mode
"""



print("Thank you for using Function Former by Making Made Easy! If you would like to support this free project and Making Made Easy, please consider becoming a member of our Patreon at https://patreon.com/MakingMadeEasy ")

import requests
import subprocess
import os
import time
import json
import threading
import traceback
import re
import sys
chat_history = []
try:
    with open('a_k.txt', 'r') as file:
        api_key = file.read()
except:
    api_key = input('Paste your OpenAI API key here: ')
    with open('a_k.txt', 'w+') as file:
        file.write(api_key)
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
code_file_path = 'generated_code.py'
code_file_path_edited = 'edited_code.py'
with open('output.log', 'w+') as file:
    file.write('')
with open('goal.txt', 'w+') as file:
    file.write('')


def display_file_contents():
    last_modified = None
    filename = 'generated_code.py'
    while True:
        try:
            current_modified = os.path.getmtime(filename)
            if last_modified is None or current_modified > last_modified:
                with open(filename, 'r') as file:
                    contents = file.read()
                print("\033[H\033[J", end="")  # Clear console screen
                print(contents)
                last_modified = current_modified
            time.sleep(1)  # Check every second
        except FileNotFoundError:
            print("File not found. Ensure the file exists.")
            time.sleep(1)  # Wait a bit before trying again
        except KeyboardInterrupt:
            print("Stopping file monitoring.")
            break


def filter_python_lines(lines):
    """ Remove lines that only contain the word 'python'. """
    return [line for line in lines if line.strip().lower() != 'python']

def modify_print_statements(lines):
    """ Modify print statements to log output to a file, ensuring each logging line starts on a new line. """
    modified_lines = []
    for line in lines:
        stripped_line = line.lstrip()
        indent = ' ' * (len(line) - len(stripped_line))

        if "print(" in stripped_line:
            # Keep the original print statement
            modified_lines.append(indent + stripped_line)
            # Extract the content inside print()
            log_content = str(stripped_line.split("print(", 1)[1].rsplit(")", 1)[0])
            # Prepare the log line with the same level of indentation and ensure it starts on a new line
            log_line = f'{indent}with open("output.log", "a") as log_file: log_file.write({log_content} + "\\n")\n'
            modified_lines.append(log_line)
        else:
            modified_lines.append(line)

    return modified_lines

def wrap_with_try_except(lines):
    """ Wrap the code in a try-except block for logging errors, ensuring all lines are correctly indented within the try block. """
    wrapped_lines = ['import traceback\n', 'try:\n']
    for line in lines:
        # Always add an additional four spaces to each line for the try block indentation
        wrapped_lines.append('    ' + line)
    wrapped_lines.extend([
        'except Exception:\n',
        '    with open("output.log", "a") as log_file:\n',
        '        traceback.print_exc(file=log_file)\n'
    ])
    return wrapped_lines

def instrument_file(input_filename, output_filename):
    # Read the input file
    with open(input_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Apply transformations
    lines = filter_python_lines(lines)
    print('adding try and except stuff')
    #lines = wrap_with_try_except(lines)
    print('add print statements and output logging')
    #lines = modify_print_statements(lines)

    # Write the modified lines to the output file, ensuring each line ends properly with a new line
    with open(output_filename, 'w', encoding='utf-8') as file:
        file.writelines(line if line.endswith('\n') else line + '\n' for line in lines)


def monitor_file_size(log_file_path, max_lines, process):
    """ Monitors the number of lines in a given log file and terminates the associated process if exceeded. """
    try:
        while process.poll() is None:  # Only monitor while the subprocess is running
            with open(log_file_path, 'r') as file:
                line_count = sum(1 for line in file)
            if line_count > max_lines:
                print(f"Log file exceeded {max_lines} lines. Terminating process.")
                process.terminate()
                break
            time.sleep(1)  # Check every second
    except FileNotFoundError:
        print("Log file not found. Please check the path and filename.")
def replace_with_correct_indentation(code, old, new):
    # This function replaces `old` with `new` in `code` while matching the leading whitespace of `old`
    pattern = re.escape(old).replace(r'\ ', r'\s+')  # Allow any whitespace to match in the 'old' pattern
    matches = list(re.finditer(pattern, code))
    
    if not matches:
        return code  # If no match is found, return original code

    # If match is found, replace ensuring the same leading whitespace is preserved
    for match in reversed(matches):  # Iterate in reverse to not mess up indices when replacing
        start, end = match.span()
        # Find the leading whitespace by looking at the start of the line to the start of the match
        leading_whitespace = re.match(r'^\s*', code[:start].split('\n')[-1]).group(0)
        indented_new = ''.join(leading_whitespace + line if index == 0 else '\n' + leading_whitespace + line 
                                for index, line in enumerate(new.splitlines()))
        code = code[:start] + indented_new + code[end:]
    
    return code
def remove_tabs(lst):
    return [item.replace('\t', '') for item in lst]
def strip_leading_whitespace(lst):
    return [item.lstrip() for item in lst]

def validate_and_run_code(goal_file):
    edit_loop_count = 0
    main_loop_count = 0
    global chat_history
    error_count = {}
    initial_request1 = input('Please give a thorough and detailed description of the function or script that you want: ')
    new_or_existing = input('Generate new code or edit an existing file (Say 1 for new or 2 for existing): ')
    if new_or_existing == '2':
        filename = input('Please put the existing python file in the same folder as this script, and then provide the filename here: ')
        og_filename = filename
        with open(filename, 'r') as file:
            code = file.read()
            
            code = "\n".join([line for line in code.split('\n') if line.strip() != ""])
    else:
        print('Creating new script')
        og_filename = None
    auto_or_chat = input("Choose 1 for Auto-Coding or 2 for Chat-Coding: ") #Chat just asks for user input each loop whereas auto skips the user input part
    process_timeout = input ("Choose 1 to have a timeout for running scripts or 2 for allowing the scripts to run until complete (potentially indefinitely if looping): ") #if you choose 2, just close the process window for the script if it gets frozen or something
    time.sleep(5)
    if initial_request1 == '':
        try:
            with open(goal_file, 'r') as file:
                initial_request = file.read().strip()
        except:
            print('no prompt given')
            
    else:
        initial_request = initial_request1
        with open(goal_file, 'w+') as file:
            file.write(initial_request)
    while True:  # MAIN LOOP
        try:
            with open(goal_file, 'r') as file:
                initial_request = file.read().strip()
            print("Main loop count: " + str(main_loop_count))
            print("Edit loop count: " + str(edit_loop_count))
            print('Prompt: ' + str(initial_request))
            if new_or_existing == '1':
                print('creating new')
                try:
                    payload = {
                        "model": "gpt-4o",
                        "messages": [
                            {"role": "system", "content": "You are a Python code assistant."},
                            {"role": "user", "content": "ACTUAL PROMPT IS EVERYTHING AFTER THIS: \n\n Please create this python script. You cannot say any words except for the script itself and do not put the word python at the beginning of the script. You need a print statement for literally everything in the script, say if stuff was successful or not, or just any pertinent data. Like print statements for when background and backend stuff is happening. It needs to have literally as many print statements as possible in all areas of the script. You can literally only respond with the script itself and no other words or sentences: " + initial_request + " \n\n You can never use placeholder logic."}
                        ]
                    }
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(payload))





                    code = str(response.json()['choices'][0]['message']['content']).strip().replace("'''", "").replace("```", "").replace('\u00B0', '')
                    cleaned_code = "\n".join([line for line in code.split('\n') if line.strip() != ""])
                    print("Code generated successfully")
                    # Write the code ensuring UTF-8 encoding
                    with open(code_file_path, 'w+', encoding='utf-8') as file:
                        file.write(cleaned_code)

                except Exception as e:
                    print(e)
                    time.sleep(1)
                    continue
            else:
                print('using script from file')

                try:
                    with open(filename,'r') as file:
                        code = file.read()
                    cleaned_code = "\n".join([line for line in code.split('\n') if line.strip() != ""])
                    # Write the code ensuring UTF-8 encoding
                    with open(code_file_path, 'w+', encoding='utf-8') as file:
                        file.write(cleaned_code)
                except:
                    pass
                


            # Process and write the instrumented file with UTF-8 encoding
            instrument_file(code_file_path, code_file_path_edited)
  


            while True:  # EDIT LOOP
                with open(code_file_path, 'r', encoding='utf-8') as file:
                    code = file.read()
                with open(goal_file, 'r') as file:
                    initial_request = file.read().strip()
                print("Main loop count: " + str(main_loop_count))
                print("Edit loop count: " + str(edit_loop_count))

                

                try:

                    with open('output.log', 'w+') as file:
                        file.write('')
                    if edit_loop_count == 0 and new_or_existing == '2':
                        pass
                    else:
                        print('running script to see if it works\n')

                        with open('output.log', 'w') as log_file:
                            try:
                                process = subprocess.Popen(
                                    ["python", code_file_path_edited],
                                    stdout=log_file,
                                    stderr=log_file,
                                    creationflags=subprocess.CREATE_NEW_CONSOLE
                                )
                     
                                if process_timeout == '1':
                                    # Start a thread to monitor the log file size
                                    monitor_thread = threading.Thread(target=monitor_file_size, args=('output.log', 5000, process))
                                    monitor_thread.start()

                                    # Wait for the process to complete
                                    process.wait(timeout=10)
                                else:
                                    process.wait()
                            except subprocess.TimeoutExpired:
                                process.terminate()  # Ensure the process is terminated after a timeout
                            except Exception as e:
                                print(f"An error occurred: {str(e)}\n")
                        print('script finished\n')
                    
                    # Read from the file and print its contents
                    time.sleep(5)
                    try:
                        with open("output.log", "r") as f:
                            lines = []
                            for i in range(200):  # Set limit to 200 lines
                                line = f.readline()
                                if not line:  # Break if the end of file is reached
                                    break
                                lines.append(line)

                            output = '\n'.join(lines)  # Join the list of lines into a single string
                       
                        match = re.search(r'File ".*", line (\d+)\n([^\n]*)\n\s*\^+\n\s*IndentationError: unexpected indent', output)

                        # If there's a module not found error, attempt to install the missing module using pip
                        if "No module named" in output:
                            missing_module = re.search(r"No module named '(\w+)'", output)
                            if missing_module:
                                module_name = missing_module.group(1)
                                # Prepare the data to send to ChatGPT

                                validation_prompt = 'Write the command to install the Python module ' + module_name + ' using pip, and nothing else. Your full response will be copy and pasted into the terminal so literally only say the command.'
                                validation_payload = {
                                    "model": "gpt-4o-mini",
                                    "messages": [
                                        {"role": "system", "content": "You are a Python code assistant."},
                                        {"role": "user", "content": validation_prompt}
                                    ]
                                }
                                validation_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(validation_payload))
                                chat_response = validation_response.json()['choices'][0]['message']['content'].strip().replace("'''", "").replace("```", "")
                                print(chat_response)
                                user = input('Install ' + str(module_name) + '? (Yes/no): ')
                                if user.lower() == 'yes' or user.lower() == 'y':
                                    print('installing ' + str(module_name))
                                    # Execute the command from ChatGPT if it is valid
                                    subprocess.call(chat_response.split(' '))
                                    subprocess.call([sys.executable, '-m', chat_response.split(' ')[0], chat_response.split(' ')[1], chat_response.split(' ')[2]])
                                    print('Finished installing ' + str(module_name))
                                else:
                                    chat_history.append('User declined install of ' + str(module_name) + '. You must choose a different module to use.')
                            
                            elif 'IndentationError: unexpected indent' in output:
                                line_number = int(match.group(1))
                                problematic_line = match.group(2)

                                print(f"IndentationError found on line {line_number}: {problematic_line}")

                                # Read the original code file
                                with open('edited_code.py', 'r') as code_file:
                                    code_lines = code_file.readlines()

                                # Check if the line from the error matches the corresponding line in the code file
                                if problematic_line.strip() == code_lines[line_number - 1].strip():
                                    # Remove one leading whitespace (either space or tab) if present
                                    if code_lines[line_number - 1].startswith((' ', '\t')):
                                        corrected_line = code_lines[line_number - 1][1:]
                                        code_lines[line_number - 1] = corrected_line
                                        
                                        # Write the corrected lines back to the file
                                        with open('edited_code.py', 'w') as code_file:
                                            code_file.writelines(code_lines)
                                        with open('generated_code.py', 'w') as code_file:
                                            code_file.writelines(code_lines)
                                        code = '\n'.join(code_lines)
                                        print(f"Line {line_number} has been fixed by removing one leading whitespace.")
                                        continue
                                    else:
                                        print(f"Line {line_number} does not start with a space or tab, no change made.")
                                else:
                                    print("The problematic line in the log does not match the code file content.")
                            else:
                                pass
                    except Exception as e:
                        print(e)

                    chat_history.append(initial_request)
                    chat_history.append(output)
                    

                    print('Confirming if code worked or not (PRESS CTRL-C TO STOP THE PROCESS)\n')
                    validation_prompt = "Im trying to do this: " + initial_request + " with the following script: \n\n " + code + " \n\n It produced this logging output when i ran the script: \n\n " + output + " \n\n If it didnt work, please tell me why it didnt work but you cant say more than one paragraph. If the code doesnt have the required stuff to do what I want it do, then tell what it needs but you cant say more than one paragraph. If it does show basically the correct info on the output and the code looks correct, then say it worked. If it is something that is supposed to run continuously, then specify if the looping is done correctly or not.  As long as the output looks like it is basically working and theres like no errors and the code looks correct then it probably is correct and you can tell chatgpt (the chatgpt that will receive your response) that it can say the word yes. If the output is empty then it did not work correctly. \n\n You can never use placeholder logic. \n\n Your response needs to only be a paragraph or less and say the problem and the definitive solution. Your response must be only a few sentences or less and you cannot provide code in your response, only sentences. If there is still more to add to the code to get it to do what i want, then you cannot say it is working yet. \n\n So your response choices are either YES or EDIT CODE. You must only say your response choice, followed by ~~ with a space on each side, followed by the reasoning for the decision or what needs to be fixed and how to fix it, but you cant say more than one paragraph. If there is an error in output, then include that info in EDIT CODE. If you see any syntax errors in the script, also mention that on your EDIT CODE. And dont forget that you are responsible for determining if code is working decently enough to say yes"
                    validation_payload = {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You are a Python code assistant."},
                            {"role": "user", "content": validation_prompt}
                        ]
                    }
                    validation_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(validation_payload))
                    validation_result = validation_response.json()['choices'][0]['message']['content'].strip().replace("'''", "").replace("```", "")
                    chat_history.append(validation_result)



                    try:
                        valid = validation_result.split(' ~~ ')
                        choice = valid[0]
                        needs = valid[1]
                    except:
                        needs = None
                        choice = "yes"
                    print(choice)

                    if choice.lower().strip() == "yes":
                        
                        keep_on = input("\n\nCode seems to have worked correctly. Do you want to change or upgrade anything? Either say no, describe what you want, or say CHAT to start a conversation with ChatGPT (Convo is used as extra context for coding): ")
                        if keep_on == 'no':
                            os._exit(0)
                        elif keep_on == 'CHAT':
                            while True:
                                try:
                                    with open(goal_file, 'r') as file:
                                        initial_request = file.read().strip()
                                except:
                                    print('no prompt given')
                                    break
                                user_input = input("Enter additional information or say END CHAT to end this chat: ")

                                if user_input.lower() == 'end chat':
                                    break
                                else:
                                    pass
                                # chatgpt response to input based on chat_history
                                prompt = "YOU MUST KEEP YOUR RESPONSE SHORTER THAN A PARAGRAPH! \n\n CHAT HISTORY: " + initial_request + " \n\n OTHER DATA: \n\n I ran into this issue: " + str(needs) + " \n\n CURRENT CODE: \n\n " + code + " \n\n ACTUAL PROMPT: " + user_input
                                payload = {
                                    "model": "gpt-4o-mini",
                                    "messages": [
                                        {"role": "system", "content": "You are a Python code assistant."},
                                        {"role": "user", "content": prompt}
                                    ]
                                }
                                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(payload))
                                result = response.json()['choices'][0]['message']['content'].strip().replace("'''", "").replace("```", "")
                                print('Response: ' + result)
                                print()

                                initial_request = initial_request+'\n\nuser input: ' + user_input
                                initial_request = initial_request+'\nchatgpt response to user input: ' + result
                                with open(goal_file, 'w+') as file:
                                    file.write(initial_request)
                        else:
                            with open(goal_file, 'w+') as file:
                                file.write(initial_request + '\n\n' + keep_on)
                            initial_request = initial_request + '\n\n User Input after ChatGPT detected code worked correctly: ' + keep_on
                            break


                    else:
                        if auto_or_chat == '2':
                            user_in = input('CURRENT CODE:\n\n'+code+'\n\nTERMINAL OUTPUT FROM RUNNING SCRIPT:\n '+output+'\n\nWhat ChatGPT says needs to happen: '+str(needs)+ '\n\nChange choice or go with decision from ChatGPT? Press Enter to go with the decision from ChatGPT, give your own description of what it needs to fix, or say STOP to end code creation: ')

                            if user_in == "STOP":
                                os._exit(0)
                            elif user_in == "":
                                pass
                            else:
                                needs = 'What Chatgpt says the script needs: ' + str(needs) + '\nUser Input about this: ' + user_in
                                edit_loop_count -= 1
                        else:
                            print(needs)
                        edit_loop_count += 1
                        
                                         
                        if edit_loop_count >= 4:
                            print('\n\n\n\n\n\nEditing still needed after 4 attempts. Regenerating code from scratch.')
                            edit_loop_count = 0
                            main_loop_count += 1
                            if main_loop_count >= 4:
                                print(('\n' * 50) + str(needs))
                  
                                print('Starting chat with user\n')
                                while True:
                                    try:
                                        with open(goal_file, 'r') as file:
                                            initial_request = file.read().strip()
                                    except:
                                        print('no prompt given')
                                        break
                                    user_input = input("Enter additional information or say END CHAT to end this chat: ")

                                    if user_input.lower() == 'end chat':
                                        break
                                    else:
                                        pass
                                    # chatgpt response to input based on chat_history
                                    prompt = "Chat History: " + initial_request + " \n\n OTHER DATA: \n\n I ran into this issue: " + str(needs) + " \n\n CURRENT CODE: \n\n " + code + " \n\n ACTUAL PROMPT: " + user_input
                                    payload = {
                                        "model": "gpt-4o-mini",
                                        "messages": [
                                            {"role": "system", "content": "You are a Python code assistant."},
                                            {"role": "user", "content": prompt}
                                        ]
                                    }
                                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(payload))
                                    result = response.json()['choices'][0]['message']['content'].strip().replace("'''", "").replace("```", "")
                                    print('Response: ' + result)
                                    print()

                                    initial_request = initial_request+'\n\nuser input: ' + user_input
                                    initial_request = initial_request+'\nchatgpt response to user input: ' + result
                                    with open(goal_file, 'w+') as file:
                                        file.write(initial_request)
                                print('Ending chat with user\n')
                                main_loop_count = 0
                                if new_or_existing == '2':
                                    filename = og_filename
                                else:
                                    pass
                                break
                            else:
                                if new_or_existing == '2':
                                    filename = og_filename
                                else:
                                    pass
                                break
                        else:
                            print('\n\n\n\n\n\nEditing needed. Getting list of necessary edits')

                        edits_prompt = "Make sure you have lots of print debug statements in any code you provide. Like basically for the success or failure of almost every line. \n\n Modify this code: \n\n " + code + " \n\n To have this stuff, abilities, and fixes: " + needs + '\n'+initial_request+ " \n\n You absolutely literally cannot say anything except the specific lines or sections of corrected code that accomplish the current goal, and they must be said in a way that follows these instructions precisely (no prefacing statements or labels) - your entire response must be in this example format: \n\n ```codeblockstart \n old section of code that is getting replaced (Must be the precise original code worded exactly like in the original script so my program can compare this to sections of the original script to find where to put the new code) \n ~~ \n correct line or lines of code that get copied and pasted into my script to make it work (You must include any original code that remains unchanged as well) \n codeblockend```\n\n```codeblockstart \n old section of code that is getting replaced (Must be the precise original code worded like in the original script so my program can compare this to sections of the original script to find where to put the new code) \n ~~ \n correct line or lines of code that get copied and pasted into my script to replace the original section of code to make it work (You must include any original code that remains unchanged as well) \n codeblockend```  \n\n\n Only provide the lines that actually need to be changed to accomplish the goal, leave everything else how it is. Like dont regenerate the whole script, just provide the areas that need to be modified. You must follow the formatting example precisely. If tabbing is incorrect, fix it."
                        edits_payload = {
                            "model": "gpt-4o",
                            "messages": [
                                {"role": "system", "content": "You are a Python code assistant."},
                                {"role": "user", "content": edits_prompt}
                            ]
                        }
                        edits_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(edits_payload))
                        edits_result = edits_response.json()['choices'][0]['message']['content']
                        print('got necessary changes from AI')
                        code_changes = edits_result.replace('```codeblockstart', '').split("codeblockend```")
                        del code_changes[len(code_changes)-1]




                        """



                        #THIS DOESNT SEEM TO HELP SO IT IS COMMENTED OUT FOR NOW



                        
                        confirm_prompt = "Make sure you have lots of print debug statements in any code you provide. Like basically for the success or failure of almost every line. \n\n ChatGPT says I should make these chages to my code: \n\n " + '\n'.join(code_changes) + " \n\n To have this stuff, abilities, and fixes: " + needs + '\n'+initial_request+ " \n\n Here is my code that the changes will happen to: \n "+code+" \n\n You absolutely literally cannot say anything except the word GOOD if the edits will work, or if they wont work then you must provide the specific lines or sections of corrected code that accomplish the current goal, and they must be said in a way that follows these instructions precisely (no prefacing statements or labels) - your entire response must be in this example format: \n\n ```codeblockstart \n old section of code that is getting replaced (Must be the precise original code worded exactly like in the original script so my program can compare this to sections of the original script to find where to put the new code) \n ~~ \n correct line or lines of code that get copied and pasted into my script to make it work (You must include any original code that remains unchanged as well) \n codeblockend```\n\n```codeblockstart \n old section of code that is getting replaced (Must be the precise original code worded like in the original script so my program can compare this to sections of the original script to find where to put the new code) \n ~~ \n correct line or lines of code that get copied and pasted into my script to replace the original section of code to make it work (You must include any original code that remains unchanged as well) \n codeblockend```  \n\n\n Only provide the lines that actually need to be changed to accomplish the goal, leave everything else how it is. Like dont regenerate the whole script, just provide the areas that need to be modified. You must follow the formatting example precisely. If tabbing is incorrect, fix it. \n\n So yeah, if the edits I provided in this prompt will work, then just say only the word GOOD, otherwise follow all these instructions. But if it looks like the edits will do basically do good enough then just say the word GOOD. The edits that I have given you in this prompt follow the same formatting as the instructions from this prompt."
                        confirm_payload = {
                            "model": "gpt-4o",
                            "messages": [
                                {"role": "system", "content": "You are a Python code assistant."},
                                {"role": "user", "content": confirm_prompt}
                            ]
                        }
                        confirm_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(confirm_payload))
                        confirm_result = str(confirm_response.json()['choices'][0]['message']['content'])
                        if confirm_result.strip().replace(' ','') == "GOOD":
                            print('Edits are good. Making edits now.')
                        else:
                            print('got necessary changes from AI')
                            code_changes = confirm_result.replace('```codeblockstart', '').split("codeblockend```")
                            del code_changes[len(code_changes)-1]
                        """





                        
                        code_index = 0
                        

                        line_amount = 0
                        while True:
                            try:
                                #print(len(code_changes))
                                if len(code_changes) == 0:

                                    validation_prompt = "Make sure you have lots of print debug statements in any code you provide. Like basically for the success or failure of almost every line. \n\n Modify this code: \n\n " + code + " \n\n To have this stuff, abilities, and fixes: " + needs + " \n\n You absolutely literally cannot say anything except the specific lines or sections of corrected code that accomplish the current goal, and they must be said in a way that follows these instructions precisely (no prefacing statements or labels) - your entire response must be in this example format: \n\n ```codeblockstart \n old section of code that is getting replaced (Must be the precise original code worded exactly like in the original script so my program can compare this to sections of the original script to find where to put the new code) \n ~~ \n correct line or lines of code that get copied and pasted into my script to make it work (You must include any original code that remains unchanged as well) \n codeblockend```\n\n```codeblockstart \n old section of code that is getting replaced (Must be the precise original code worded like in the original script so my program can compare this to sections of the original script to find where to put the new code) \n ~~ \n correct line or lines of code that get copied and pasted into my script to replace the original section of code to make it work (You must include any original code that remains unchanged as well) \n codeblockend```  \n\n\n Only provide the lines that actually need to be changed to accomplish the goal, leave everything else how it is. Like dont regenerate the whole script, just provide the areas that need to be modified. You must follow the formatting example precisely. If tabbing is incorrect, fix it."
                                    validation_payload = {
                                        "model": "gpt-4o",
                                        "messages": [
                                            {"role": "system", "content": "You are a Python code assistant."},
                                            {"role": "user", "content": validation_prompt}
                                        ]
                                    }
                                    validation_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(validation_payload))
                                    validation_result = validation_response.json()['choices'][0]['message']['content']
                                    code_changes = validation_result.replace('```codeblockstart', '').split("codeblockend```")
                                    del code_changes[len(code_changes)-1]
                                    continue
                                else:
                                    pass


                                    
                                
                                try:
                                    current_change = code_changes[code_index].split('~~')
                                except:
                                    code_changes = []
                                
                                

                                old_lines_of_code = "\n".join([line for line in current_change[0].replace("```", "").split('\n') if line.strip() != ""])
                                new_lines_of_code = "\n".join([line for line in current_change[1].replace("```", "").split('\n') if line.strip() != ""])
                                old_code_list = old_lines_of_code.split('\n')
                                new_code_list = new_lines_of_code.split('\n')
                                code_list = code.split('\n')
                                # Iterate through the code to find a segment that matches the old_code_list
                                start_index = -1
                                for idx in range(len(code_list) - len(old_code_list) + 1):
                                    segment = code_list[idx:idx + len(old_code_list)]
                                    adjusted_segment = remove_tabs(segment)
                                    adjusted_old_code_list = remove_tabs(old_code_list)
                                    #print('\n\nOLD CODE 2:\n'+'\n'.join(adjusted_old_code_list))
                                    
                                    adjusted_segment2 = strip_leading_whitespace(adjusted_segment)
                                    adjusted_old_code_list2 = strip_leading_whitespace(adjusted_old_code_list)
                                    #print('\n\nSEGMENT FROM FILE:\n'+'\n'.join(adjusted_segment2).strip().replace(' ',''))
                                    #print('\n\nADJUSTED OLD CODE:\n'+'\n'.join(adjusted_old_code_list2).strip().replace(' ',''))
                                    
                                    if str(adjusted_old_code_list2).strip().replace(' ','') == str(adjusted_segment2).strip().replace(' ',''):
                                        # Found a match, now adjust new_code_list indentation
                                        start_index = idx
                      

                                        # Determine the number of leading spaces in the original segment
                                        leading_spaces = len(segment[0]) - len(segment[0].lstrip())

                                        # Determine the minimum existing indentation in new_code_list
                                        min_new_indent = min(len(line) - len(line.lstrip()) for line in new_code_list if line.strip())

                                        # Adjust new_code_list indentation
                                        indented_new_code = [
                                            ' ' * (leading_spaces + max(0, len(line) - len(line.lstrip()) - min_new_indent)) + line.lstrip()
                                            for line in new_code_list
                                        ]

                                    
                                        """
                                        print('\n\nFound and adjusted match at index:', start_index)
                                        print('\n\nSEGMENT FROM FILE:\n'+'\n'.join(segment))
                                        print('NEW CODE:\n'+'\n'.join(indented_new_code))
                                        """
                                        break

                               
                                
                                
                                code_index += 1

                                #yo = input('\n\n\n\nPress Enter if this looks correct or say what\'s wrong with this fix if it looks wrong.')
                                if start_index == -1:
                                    if code_index >= len(code_changes):
                                        break
                                    else:
                                        continue

                                # Remove old lines and prepare space for new lines
                                del code_list[start_index:start_index + len(old_code_list)]  # Delete old lines
                                code_list[start_index:start_index] = ['\n' * len(indented_new_code)]  # Insert empty new lines

                                # Replace empty new lines with actual new lines
                                code_list[start_index:start_index + len(indented_new_code)] = indented_new_code

                                if code_index >= len(code_changes):
                                    break
                                else:
                                    continue
                            except Exception as e:
                                print(current_change)
                                print(traceback.format_exc())
                                time.sleep(10)
                                code_index += 1
                                if code_index >= len(code_changes):
                                    break
                                else:
                                    continue
                        code = "\n".join([line for line in code_list if line.strip() != ""])
                        with open(code_file_path, 'w+') as file:
                            file.write(code.replace('\u00B0', ''))
                        instrument_file(code_file_path, code_file_path_edited)
                except Exception as e:
                    print('error in this script now')
                    print(traceback.format_exc())
                    error_count = {}
                    break
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            time.sleep(120)  # Wait and retry after a delay in case of exceptions
        else:
            time.sleep(0.1)
    time.sleep(5)

# Example usage:
print('starting')
validate_and_run_code('goal.txt')
