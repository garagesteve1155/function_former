



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
start_time = str(int(time.time()))
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
code_file_path = 'generated_code'+start_time+'.py'
with open('output.log', 'w+') as file:
    file.write('')
with open('goal.txt', 'w+') as file:
    file.write('')

def filter_python_lines(lines):
    """ Remove lines that only contain the word 'python'. """
    return [line for line in lines if line.strip().lower() != 'python']


def instrument_file():
    # Read the input file
    with open('generated_code'+start_time+'.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Apply transformations
    lines = filter_python_lines(lines)

    # Write the modified lines to the output file, ensuring each line ends properly with a new line
    with open('generated_code'+start_time+'.py', 'w', encoding='utf-8') as file:
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
def remove_all_indentation(lines):
    return [line.strip() for line in lines]

# Check and adjust the indentation of new lines
def remove_single_space_indent(lines):
    return [line[1:] if line.startswith(' ') and not line.startswith('  ') else line for line in lines]
def validate_and_run_code(goal_file):
    edit_loop_count = 0
    main_loop_count = 0
    global chat_history
    error_count = {}
    initial_request1 = input('Please give a thorough and detailed description of the function or script that you want: ')
    new_or_existing = input('Generate new code or edit an existing file (Say 1 for new or 2 for existing): ')
    pin_or_whole = input('Pinpoint editing or whole code regeneration editing (Say 1 for pinpoint or 2 for whole code): ')
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
            instrument_file()
  


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
                                    ["python", code_file_path],
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
                                    continue
                                else:
                                    chat_history.append()
                                    initial_request = initial_request+'\n\nUser declined install of ' + str(module_name) + '. You must choose a different module to use.'
                                    with open(goal_file, 'w+') as file:
                                        file.write(initial_request)
                            elif 'unexpected indent' in output:
                                match = re.search(r'File ".*", line (\d+)\n([^\n]*)\n\s*\^+\n\s*IndentationError: unexpected indent', output)
                                line_number = int(match.group(1))
                                problematic_line = match.group(2)

                                print(f"IndentationError found on line {line_number}: {problematic_line}")

                                # Read the original code file
                                with open('generated_code'+start_time+'.py', 'r') as code_file:
                                    code_lines = code_file.readlines()

                                # Check if the line from the error matches the corresponding line in the code file
                                if problematic_line.strip() == code_lines[line_number - 1].strip():
                                    # Remove one leading whitespace (either space or tab) if present
                                    if code_lines[line_number - 1].startswith((' ', '\t')):
                                        corrected_line = code_lines[line_number - 1][1:]
                                        code_lines[line_number - 1] = corrected_line
                                        
                                        
                                        with open('generated_code'+start_time+'.py', 'w') as code_file:
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

     
                    

                    print('Confirming if code worked or not (PRESS CTRL-C TO STOP THE PROCESS)\n')
                    if pin_or_whole == '1':
                        validation_prompt = "Im trying to do this: " + initial_request + " with the following script: \n\n " + code + " \n\n It produced this logging output when i ran the script: \n\n " + output + " \n\n If the code looks and seems like it works, only say the word YES. If it didnt work, then follow these instructions: \n Make sure you have lots of print debug statements in any code you provide. Like basically for the success or failure of almost every line. \n\n Modify this code: \n\n " + code + " \n\n To have this stuff, abilities, and fixes (YOUR SOLE PURPOSE RIGHT NOW IS TO PROVIDE ALL THE NECESSARY EDITS TO THE CODE TO HAVE THIS STUFF): " + initial_request + " \n\n This is the terminal output from the last time this script was executed so fix any issues you see from this output: \n" + output + " \n\n\n You absolutely literally cannot say anything except the specific lines or sections of corrected code that accomplish the current goal, and they must be said in a way that follows these instructions precisely (no prefacing statements or labels) - your entire response must be in this example format: \n\n ```codeblockstart \n old section of code that my program will automatically search for (Must be the precise original code worded exactly like in the original script so my program can compare this to sections of the original script to find where to put the new code. You absolutely must indent this correctly cause indentions are taken into account in my searching system.) \n ~~ \n correct line or lines of code that get copied and pasted into my script to make it work (You must include any original code that remains unchanged as well) \n codeblockend```\n\n```codeblockstart \n old section of code that my program will automatically search for (Must be the precise original code worded like in the original script so my program can compare this to sections of the original script to find where to put the new code. You absolutely must indent this correctly cause indentions are taken into account in my searching system.) \n ~~ \n correct line or lines of code that get copied and pasted into my script to replace the original section of code to make it work (You must include any original code that remains unchanged as well) \n codeblockend```  \n\n\n Only provide the lines that actually need to be changed to accomplish the goal, leave everything else how it is. Like dont regenerate the whole script, just provide the areas that need to be modified. You must follow the formatting example precisely. If tabbing is incorrect, fix it. If changes are necessary, your list of edits must include every necessary change. \n\n You cannot provide the complete script, you must only provide the specific edits. And take into account that the edits are done in the order that you give so if a previous edit includes changing a line that you have to look for in the next edit, then compensate for that on the lines that you search for on the next edit by including what it would be after the previous edit. \n\n And if you think the code is already working properly then just say the word YES. \n\n And remember to follow this formatting template if edits need to be made: \n\n ```codeblockstart \n old section of code that my program will automatically search for (Must be the precise original code worded exactly like in the original script so my program can compare this to sections of the original script to find where to put the new code. You absolutely must indent this correctly cause indentions are taken into account in my searching system.) \n ~~ \n correct line or lines of code that get copied and pasted into my script to make it work (You must include any original code that remains unchanged as well) \n codeblockend```\n\n```codeblockstart \n old section of code that my program will automatically search for (Must be the precise original code worded like in the original script so my program can compare this to sections of the original script to find where to put the new code. You absolutely must indent this correctly cause indentions are taken into account in my searching system.) \n ~~ \n correct line or lines of code that get copied and pasted into my script to replace the original section of code to make it work (You must include any original code that remains unchanged as well) \n codeblockend```  \n\n\n You can only edit up to 5 lines of code per index on your list of edits to make, but you can make unlimited total edits in your list."
                    else:
                        validation_prompt = "Im trying to do this: " + initial_request + " with the following script: \n\n " + code + " \n\n It produced this logging output when i ran the script: \n\n " + output + " \n\n If the code looks and seems like it works, only say the word YES. If it didnt work, then provide the absolutely complete fixed script with full working logic and no placeholders for anything. \n Make sure you have lots of print debug statements in any code you provide. Like basically for the success or failure of almost every line. \n\n Modify this code: \n\n " + code + " \n\n To have this stuff, abilities, and fixes: " + initial_request + " \n\n This is the terminal output from the last time this script was executed so fix any issues you see from this output: \n" + output + " \n\n\n You absolutely literally cannot say anything except the word YES if it worked, or say only the entire script with no other words or comments or placeholders."
                    validation_payload = {
                        "model": "gpt-4o",
                        "messages": [
                            {"role": "system", "content": "You are a Python code assistant."},
                            {"role": "user", "content": validation_prompt}
                        ]
                    }
                    validation_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(validation_payload))
                    validation_result = validation_response.json()['choices'][0]['message']['content'].strip().replace("'''", "").replace("```", "")
                


     
                    

                    if validation_result.lower().strip() == "yes":
                        
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
                                prompt = "YOU MUST KEEP YOUR RESPONSE SHORTER THAN A PARAGRAPH! \n\n Chat History: " + initial_request + " \n\n CURRENT CODE: \n\n " + code + " \n\n The code is doing this: " + str(output) + " \n\n ACTUAL PROMPT: " + user_input
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
                            continue
                        else:
                            with open(goal_file, 'w+') as file:
                                file.write(initial_request + '\n\n' + keep_on)
                            initial_request = initial_request + '\n\n User Input after ChatGPT detected code worked correctly: ' + keep_on
                            break


                    else:
                        if pin_or_whole == '1':
                            print('got necessary changes from AI')
                            try:
                                code_changes = validation_result.replace('```codeblockstart', '').replace('```', '').replace('codeblockstart', '').replace('\n ','\n').replace('~~ ','~~').replace('   ','    ').replace('\n\n','').replace('\n\n','').replace('\n\n','').split("codeblockend")
                            except:
                                print(traceback.format_exc())



                            edit_loop_count += 1
                            
                                             
                            if edit_loop_count >= 2:
                                print('\n\n\n\n\n\nEditing still needed after 4 attempts. Regenerating code from scratch.')
                                edit_loop_count = 0
                                main_loop_count += 1
                                if main_loop_count >= 6:
                                    main_loop_count = 0
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
                                        prompt = "YOU MUST KEEP YOUR RESPONSE SHORTER THAN A PARAGRAPH! \n\n Chat History: " + initial_request + " \n\n CURRENT CODE: \n\n " + code + " \n\n The code is doing this: " + str(output) + " \n\n ACTUAL PROMPT: " + user_input
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
                                    
                                    
                                    break
                                else:
                                    
                                    break
                            else:
                                print('\n\n\n\n\n\nEditing needed.')


                            if auto_or_chat == '2':
                                user_in = input('CURRENT CODE:\n\n'+code+'\n\nTERMINAL OUTPUT FROM RUNNING SCRIPT:\n '+output+'\n\nWhat ChatGPT says needs to happen: \n'+'\n'.join(code_changes)+ '\n\n Press Enter to go with the decision from ChatGPT, say CHAT to talk to ChatGPT, or say STOP to end code creation: ')

                                if user_in == "STOP":
                                    os._exit(0)
                                elif user_in == "":
                                    pass
                                elif user_in == "CHAT":
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
                                        prompt = "YOU MUST KEEP YOUR RESPONSE SHORTER THAN A PARAGRAPH! \n\n Chat History and code editing goal: " + initial_request + " \n\n CURRENT CODE: \n\n " + code + " \n\n The code is doing this: " + str(output) + " \n\n ChatGpt is suggesting this list of lines to look for and what to replace them with to make the code do what I want: \n "+'\n'.join(code_changes)+" \n\n ACTUAL PROMPT: " + user_input
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
                                    continue
                                else:
                                    initial_request = initial_request + '\n\n' + user_in
                                    with open(goal_file, 'w+') as file:
                                        file.write(initial_request)
                                    continue
                            else:
                                pass

                                        
                            code_index = 0

                            while True:
                                try:


                                    try:
                                        current_change = code_changes[code_index].split('~~')
                                        print("\n\n\nCURRENT CHANGE:")
                                        print(current_change)
                                        print()
                                        
                                    except Exception as e:
                                        print(traceback.format_exc())
                                        current_change = []
                                    if current_change == [] or current_change == [""] or current_change == ['']:
                                        print('nothing in current change')
                                        code_index += 1
                                        if code_index >= len(code_changes):
                                            break
                                        else:
                                            continue
                                    else:
                                        pass
                                    old_lines_of_code = "\n".join([line for line in current_change[0].replace("```", "").split('\n') if line.strip() != ""])
                                    new_lines_of_code = "\n".join([line for line in current_change[1].replace("```", "").split('\n') if line.strip() != ""])
                                    old_code_list = remove_single_space_indent(old_lines_of_code.split('\n'))
                                    new_code_list = remove_single_space_indent(new_lines_of_code.split('\n'))
                                    code_list = code.split('\n')

                           

                                        
                                    
                                    

                                    # Find where the old code matches, ignoring leading whitespaces
                                    start_index = -1
                                    for idx in range(len(code_list) - len(old_code_list) + 1):
                                        segment = remove_single_space_indent(code_list[idx:idx + len(old_code_list)])

                                        n_seg = remove_all_indentation(segment)
                                        n_old = remove_all_indentation(old_code_list)
                                        if n_seg == n_old:
                                    
                                            start_index = idx
                                            break

                                    if start_index != -1:
                                        # Remove the old code lines
                                        del code_list[start_index:start_index + len(old_code_list)]
                                        print('\n\n\n\n\n\n\n\nCODE FROM FILE:\n'+'\n'.join(segment))
                                        print('\n\nREPLACING:\n'+'\n'.join(old_code_list))
                                        print('\n\nWITH:\n'+'\n'.join(new_code_list))
                                        # Insert the new lines of code at the same index
                                        for i, line in enumerate(new_code_list):
                                            code_list.insert(start_index + i, line)

                                    code_index += 1
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




                            


                                        
                            print('EDITING FINISHED')
                            code = "\n".join([line for line in code_list if line.strip() != ""])
                            with open(code_file_path, 'w+') as file:
                                file.write(code.replace('\u00B0', ''))
                            instrument_file()
                        else:
                            print("Generating edited code")
                            #whole code regen section


                       


                            if auto_or_chat == '2':
                                user_in = input('CURRENT CODE:\n\n'+code+'\n\nTERMINAL OUTPUT FROM RUNNING SCRIPT:\n '+output+'\n\nWhat ChatGPT says needs to happen: \n'+validation_result+ '\n\n Press Enter to go with the decision from ChatGPT, say CHAT to talk to ChatGPT, or say STOP to end code creation: ')

                                if user_in == "STOP":
                                    os._exit(0)
                                elif user_in == "":
                                    pass
                                elif user_in == "CHAT":
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
                                        prompt = "YOU MUST KEEP YOUR RESPONSE SHORTER THAN A PARAGRAPH! \n\n Chat History and code editing goal: " + initial_request + " \n\n CURRENT CODE: \n\n " + code + " \n\n The code is doing this: " + str(output) + " \n\n ChatGpt is suggesting this code to do what I want: \n "+validation_result+" \n\n ACTUAL PROMPT: " + user_input
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
                                    continue
                                else:
                                    edit_loop_count -= 1
                            else:
                                pass
                            edit_loop_count += 1
                            
                                             
                            if edit_loop_count >= 4:
                                print('\n\n\n\n\n\nEditing still needed after 4 attempts. Regenerating code from scratch.')
                                edit_loop_count = 0
                                main_loop_count += 1
                                if main_loop_count >= 4:
                                    main_loop_count = 0
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
                                        prompt = "YOU MUST KEEP YOUR RESPONSE SHORTER THAN A PARAGRAPH! \n\n Chat History: " + initial_request + " \n\n CURRENT CODE: \n\n " + code + " \n\n The code is doing this: " + str(output) + " \n\n ACTUAL PROMPT: " + user_input
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
                                    
                                    
                                    break
                                else:
                                    
                                    break
                            else:
                                print('\n\n\n\n\n\nEditing needed.')

                            code = "\n".join([line for line in validation_result.split('\n') if line.strip() != ""])
                            with open(code_file_path, 'w+') as file:
                                file.write(code.replace('\u00B0', ''))
                            instrument_file()


                            
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
