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
def handle_user_input(code):
    global chat_history
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
        prompt = "Im trying to do this: " + initial_request + " \n\n CURRENT CODE: \n\n " + code + " \n\n CHAT HISTORY (oldest to newest): \n\n " + "\n".join(chat_history) + " \n\n ACTUAL PROMPT: " + user_input
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

        chat_history.append('user input: ' + user_input)
        chat_history.append('chatgpt response to user input: ' + result)
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
    lines = wrap_with_try_except(lines)
    lines = modify_print_statements(lines)

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
    global chat_history
    error_count = {}
    initial_request1 = input('Please give a thorough and detailed description of the function or script that you want: ')
    new_or_existing = input('Generate new code or edit an existing file (Say 1 for new or 2 for existing): ')
    if new_or_existing == '2':
        filename = input('Please put the existing python file in the same folder as this script, and then provide the filename here: ')
        with open(filename, 'r') as file:
            code = file.read()
            
            code = "\n".join([line for line in code.split('\n') if line.strip() != ""])
    else:
        print('Creating new script')
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


            print('Prompt: ' + str(initial_request))
            if new_or_existing == '1':
                print('creating new')
                try:
                    payload = {
                        "model": "gpt-4o",
                        "messages": [
                            {"role": "system", "content": "You are a Python code assistant."},
                            {"role": "user", "content": "If a method, technique, or URL has failed a lot in the chat history, you cannot use it again. CHAT HISTORY (oldest to newest): \n\n " + "\n".join(chat_history) + " \n\n ACTUAL PROMPT IS EVERYTHING AFTER THIS: \n\n Please create this python script. You cannot say any words except for the script itself and do not put the word python at the beginning of the script. You need a print statement for literally everything in the script, say if stuff was successful or not, or just any pertinent data. Like print statements for when background and backend stuff is happening. It needs to have literally as many print statements as possible in all areas of the script. You can literally only respond with the script itself and no other words or sentences: " + initial_request + " \n\n You can never use placeholder logic."}
                        ]
                    }
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(payload))





                    code = str(response.json()['choices'][0]['message']['content']).strip().replace("'''", "").replace("```", "").replace('\u00B0', '')
                    cleaned_code = "\n".join([line for line in code.split('\n') if line.strip() != ""])

                    print(cleaned_code)
                    # Write the code ensuring UTF-8 encoding
                    with open(code_file_path, 'w+', encoding='utf-8') as file:
                        file.write(cleaned_code)
                except Exception as e:
                    print(e)
                    time.sleep(1)
                    continue
            else:
                print('using script from file')




            # Process and write the instrumented file with UTF-8 encoding
            instrument_file(code_file_path, code_file_path_edited)
            with open(code_file_path_edited, 'r', encoding='utf-8') as file:
                edited_code = file.read()


            loop_count = 0
            while True:  # EDIT LOOP
                try:
                    with open(filename,'r') as file:
                        code = file.read()
                except:
                    pass
                cleaned_code = "\n".join([line for line in code.split('\n') if line.strip() != ""])

                
                try:

                    with open('output.log', 'w+') as file:
                        file.write('')
                    if loop_count == 0 and new_or_existing == '2':
                        pass
                    else:
                        print('running script')
                        try:

                            # Launch the subprocess in a new console window
                            process = subprocess.Popen(["python", code_file_path_edited],
                                                       creationflags=subprocess.CREATE_NEW_CONSOLE)

                            # Start a thread to monitor the log file size
                            monitor_thread = threading.Thread(target=monitor_file_size, args=('output.log', 1000, process))
                            monitor_thread.start()

                            # Wait for the process to complete
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.terminate()  # Ensure the process is terminated after a timeout
                        except Exception as e:
                            print(f"An error occurred: {str(e)}")
                        print('script finished')
                    loop_count = 1
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

                            output = ''.join(lines)  # Join the list of lines into a single string

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
                    except Exception as e:
                        print(e)

                    chat_history.append(initial_request)
                    chat_history.append(output)
                    

                    
                    validation_prompt = "Im trying to do this: " + initial_request + " with the following script: \n\n " + code + " \n\n It produced this logging output when i ran the script: \n\n " + output + " \n\n If it didnt work, please tell me why it didnt work. If the code doesnt have required stuff to do want im trying to make it do, then tell what it needs. If it does show basically the correct info on the output, then say it worked. If it is something that is supposed to run continuously, then specify if the looping is done correctly or not.  As long as the output looks like it is basically working and theres like no errors then you can tell chatgpt (the chatgpt that will receive your response) that it can say the word yes. If chatgpt says some kind of file needs to be downloaded or it needs some info from the user that it isnt able to obtain on its own like an api key or a url is repeatedly wrong or 404 or files need to be downloaded, then tell the next chatgpt to Get User Input.  If the output is empty then it did not work correctly. \n\n You can never use placeholder logic. \n\n Your response needs to only be a paragraph or less and say the problem and the definitive solution. If a method, technique, or URL has failed a lot in the chat history, you cannot use it again. If you keep seeing the same error repeated in the history, then you should not use that api or module or library or way of doing something. Your response must be only a few sentences or less. If there is still more to add to the code to get it to do what i want, then you cannot say it is working yet. \n\n So your response choices are either YES, GET USER INPUT, or EDIT CODE. You must only say your response choice, followed by ~~ with a space on each side, followed by the reasoning for the decision (aka a paragraph about what needs to be changed or what input is needed from user). If there is an error in output, then include that info in EDIT CODE. If you see any syntax errors in the script, also mention that on your EDIT CODE."
                    validation_payload = {
                        "model": "gpt-4o",
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

                    if choice.lower().strip() == "yes":
                        
                        keep_on = input("\n\nCode seems to have worked correctly. Do you want to change or upgrade anything? Either say no, describe what you want, or say CHAT to start a conversation with ChatGPT (Convo is used as extra context for coding): ")
                        if keep_on == 'no':
                            os._exit(0)
                        elif keep_on == 'CHAT':
                            handle_user_input(code)
                        else:
                            with open(goal_file, 'w+') as file:
                                file.write(initial_request + '\n\n' + keep_on)
                            initial_request = initial_request + '\n\n' + keep_on
                            break

                    elif choice.lower().strip() == 'get user input':
                        user_input = input(('\n' * 50) + str(needs) + '.\n\nLet me know when this has been taken care of by either pressing Enter or provide necessary additional information here, or say the word CHAT and we can talk more about it and you can ask me questions or give advice before we try again: ')
                        if user_input != '':
                            if user_input == 'CHAT':
                                handle_user_input(code)
                            else:
                                print('adding user input to chat history')
                                chat_history.append(user_input)
                                with open(goal_file, 'w+') as file:
                                    file.write(str(initial_request) + '\n' + str(user_input))
                                print('saved to file')
                                break
                    else:
                        print('editing section')
                        validation_prompt = "Make sure you have lots of print debug statements in any code you provide. Like basically for the success or failure of almost every line. \n\n Modify this code: \n\n " + '\n'.join(code) + " \n\n To have this stuff, abilities, and fixes: " + validation_result + " \n\n You absolutely literally cannot say anything except the specific lines or sections of corrected code that accomplish the current goal, and they must be said in a way that follows these instructions precisely (no prefacing statements or labels) - your entire response must be in this example format: \n\n ```codeblockstart \n old section of code that is getting replaced (Must be the precise original code worded exactly like in the original script so my program can compare this to sections of the original script to find where to put the new code) \n ~~ \n correct line or lines of code that get copied and pasted into my script to make it work (You must include any original code that remains unchanged as well) \n codeblockend```\n\n```codeblockstart \n old section of code that is getting replaced (Must be the precise original code worded like in the original script so my program can compare this to sections of the original script to find where to put the new code) \n ~~ \n correct line or lines of code that get copied and pasted into my script to replace the original section of code to make it work (You must include any original code that remains unchanged as well) \n codeblockend```  \n\n\n Only provide the lines that actually need to be changed to accomplish the goal, leave everything else how it is. Like dont regenerate the whole script, just provide the areas that need to be modified. You must follow the formatting example precisely. If tabbing is incorrect, fix it."
                        validation_payload = {
                            "model": "gpt-4o",
                            "messages": [
                                {"role": "system", "content": "You are a Python code assistant."},
                                {"role": "user", "content": validation_prompt}
                            ]
                        }
                        validation_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(validation_payload))
                        validation_result = validation_response.json()['choices'][0]['message']['content']
                        print('got necessary changes from AI')
                        code_changes = validation_result.replace('```codeblockstart', '').split("codeblockend```")
                        del code_changes[len(code_changes)-1]
                        code_index = 0
                        if new_or_existing == '2':
                            with open(filename, 'r') as file:
                                code = file.readlines()
                        else:
                            pass

                        line_amount = 0
                        while True:
                            try:
                                #print(len(code_changes))
                                if len(code_changes) == 0:

                                    validation_prompt = "Make sure you have lots of print debug statements in any code you provide. Like basically for the success or failure of almost every line. \n\n Modify this code: \n\n " + '\n'.join(code) + " \n\n To have this stuff, abilities, and fixes: " + validation_result + " \n\n You absolutely literally cannot say anything except the specific lines or sections of corrected code that accomplish the current goal, and they must be said in a way that follows these instructions precisely (no prefacing statements or labels) - your entire response must be in this example format: \n\n ```codeblockstart \n old section of code that is getting replaced (Must be the precise original code worded exactly like in the original script so my program can compare this to sections of the original script to find where to put the new code) \n ~~ \n correct line or lines of code that get copied and pasted into my script to make it work (You must include any original code that remains unchanged as well) \n codeblockend```\n\n```codeblockstart \n old section of code that is getting replaced (Must be the precise original code worded like in the original script so my program can compare this to sections of the original script to find where to put the new code) \n ~~ \n correct line or lines of code that get copied and pasted into my script to replace the original section of code to make it work (You must include any original code that remains unchanged as well) \n codeblockend```  \n\n\n Only provide the lines that actually need to be changed to accomplish the goal, leave everything else how it is. Like dont regenerate the whole script, just provide the areas that need to be modified. You must follow the formatting example precisely. If tabbing is incorrect, fix it."
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
                                if isinstance(code, list):
                                    try:
                                        code = '\n'.join(code)
                                    except Exception as e:
                                        print(f"Error joining code list: {e}")
                                code = "\n".join([line for line in code.split('\n') if line.strip() != ""])

                                old_lines_of_code = "\n".join([line for line in current_change[0].replace("```", "").split('\n') if line.strip() != ""])
                                new_lines_of_code = "\n".join([line for line in current_change[1].replace("```", "").split('\n') if line.strip() != ""])
                                old_code_list = old_lines_of_code.split('\n')
                                new_code_list = new_lines_of_code.split('\n')
                                code_list = code.split('\n')  # 'code' should be the current content of the file
                                # Iterate through the code to find a segment that matches the old_code_list after adjustment
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

                                    
                                        
                                        print('\n\nFound and adjusted match at index:', start_index)
                                        print('\n\nSEGMENT FROM FILE:\n'+'\n'.join(segment))
                                        print('NEW CODE:\n'+'\n'.join(indented_new_code))
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
                        with open(code_file_path, 'w+') as file:
                            try:
                                code = '\n'.join(code_list)
                            except:
                                pass
                            cleaned_code = "\n".join([line for line in code.split('\n') if line.strip() != ""])
                            
                            file.write(cleaned_code.replace('\u00B0', ''))
                            filename = code_file_path
                            new_or_existing = '2'
              
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
