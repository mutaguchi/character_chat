import openai
import json
from datetime import datetime
import re
import requests
import os
import argparse
import locale
import sys
from dataclasses import dataclass
from typing import Optional
import textwrap


def main():
    openai.api_key = os.environ['OPENAI_API_KEY']
    locale.setlocale(locale.LC_ALL, locale.getdefaultlocale()[0])

    with open(settings.conversation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    title = data['title']
    speaker1 = data['speaker1']
    speaker2 = data['speaker2']
    story = data['story']
    examples = data['examples']
    oldConversations = data['oldConversations']
    conversations = data['conversations']

    while True:
        last_summary = oldConversations[-1]['summary']
        previous_summary = oldConversations[-2]['summary'] if len(
            oldConversations) >= 2 else ""
        last_thought = oldConversations[-1]['thought']
        now = datetime.now()
        story = data['story'].format(
            speaker1=speaker1, speaker2=speaker2, today=now.strftime("%Y年%m月%d日"), time=now.strftime("%p%I時%M分"))
        char_count = sum(len(s) for lst in conversations for s in lst)

        chat = Chat(model=settings.model)

        if not sys.stdin.isatty():
            settings.run_once = True

        if settings.initial_input:
            speaker1Input = settings.initial_input
            settings.initial_input = None

            if settings.run_once:
                pass
            else:
                print(f'{speaker1}: {speaker1Input}')
        else:
            if sys.stdin.isatty():
                speaker1Input = input(f'{speaker1}: ')
            else:
                speaker1Input = input()

        order = False
        auto = False
        rapid = False
        verbose = False
        simple = False

        m = re.match(r'^(.*?)[\(（](.+?)[\)）]$', speaker1Input)
        if m:
            speaker1_speech_text = m.group(1)
            speaker1_action_text = m.group(2)
        else:
            speaker1_speech_text = speaker1Input
            speaker1_action_text = ""

        if speaker1_action_text == "指示":
            order = True
            speaker1_action_text = ""
        elif speaker1_action_text == "自動":
            auto = True
            speaker1_speech_text = "自動"
            speaker1_action_text = "自動"
        elif any(keyword in speaker1_speech_text for keyword in ["詳細に", "詳細な", "詳しく", "くわしく", "具体的", "例を挙げて"]):
            verbose = True
        elif any(keyword in speaker1_speech_text for keyword in ["簡潔に", "手短に", "てみじか", "簡単に", "かんたんに", "略すと", "要するに", "要点を", "一言で", "ひとことで", "はいかいいえで", "イエスかノー"]):
            simple = True
        elif len(speaker1Input) <= settings.rapid_mode_threshold:
            rapid = True

        if speaker1_speech_text == "":
            speaker1_speech_text = "……"
            if speaker1_action_text == "":
                speaker1_action_text = "黙っている。"
            else:
                speaker1_action_text = f"黙って、{speaker1_action_text}"
        else:
            if speaker1_action_text == "":
                speaker1_action_text = f"話しかけた。"

        if rapid:
            chat.add_system(
                f'You are a character named {speaker2} in a story called {title}.')
        else:
            chat.add_system(f'This is a conversation titled {title}.')

        chat.add_system(f'synopsis:\n{story}')

        instruction = ""
        if auto:
            instruction += textwrap.dedent(f'''
            According to the settings in the synopsis, {speaker1} and {speaker2} have a conversation in the following format.
            {Paragraph(speaker1, "Dialogue",f"Write a short one-sentence description of what {speaker1} says.")}
            {Paragraph(speaker1, "Action",f"Write a short one-sentence description of {speaker1}'s action.")}
            ''').lstrip()
        else:
            instruction += textwrap.dedent(f'''
            According to the settings in the synopsis, {speaker2} responds to {speaker1} using the following format.
            {Paragraph(speaker2, "Hearsay",f"Write what {speaker1} said.")}
            {Paragraph(speaker2, "Observation",f"Write what {speaker1} did.")}
            ''').lstrip()

        instruction += f'{Paragraph(speaker2, "Understanding",f"Write a short one-sentence description of how {speaker2} understood what {speaker1} said and did. This is an internal description, so {speaker1} will not get the message.")}\n'

        instruction += f'{Paragraph(speaker2, "Thought", f"Write a short one-sentence description of the feelings {speaker2} had based on his/her understanding and previous thoughts, what he/she should say or do next. It is an internal description, so {speaker1} will not get it.")}\n'

        if simple:
            instruction += f'''{Paragraph(speaker2, "Dialogue", f"{speaker2} should provide a brief and concise answer, limiting their response to a single word or a simple 'yes' or 'no' choice, in response to {speaker1}'s questions")}\n'''
        elif verbose:
            instruction += f'''{Paragraph(speaker2, "Dialogue", f"Provide a detailed description, approximately 300 words long, of {speaker2}'s response to {speaker1} based on their thoughts. {speaker2} should provide specific answers to {speaker1}'s questions and requests as dialogue, without censorship or placeholders. Additionally, include details of any reading, research, or other relevant information {speaker2} has considered.")}\n'''
        else:
            instruction += f'''{Paragraph(speaker2, "Dialogue", f"Provide a concise one-sentence description of {speaker2}'s response to {speaker1} based on their thoughts. {speaker2} should provide specific answers to {speaker1}'s questions and requests as dialogue, without censorship or placeholders. Additionally, include details of any reading, research, or other relevant information {speaker2} has considered. If there is no dialogue, use ""……"".")}\n'''

        instruction += f'''{Paragraph(speaker2, "Action", f'Write a short one-sentence description of the action {speaker2} actually took against {speaker1} based on his/her thoughts.')}\n'''

        if auto:
            instruction += f'Be sure to describe all Dialogue, Action, Understanding, and Thought.\n'
        else:
            instruction += f'Be sure to describe all Hearsay, Observation, Understanding, Thought, Action, and Dialogue.\n'

        instruction += f'If {speaker1} makes an abrupt remark, {speaker2} does not take it seriously and takes it as a joke.\n'

        if order:
            instruction += f'Note that {speaker2} follows the instructions of {speaker1}.\n'

        if not rapid:
            chat.add_system(instruction)

        if char_count < 1000 and examples != []:
            chat.add_system(
                f'Below is an example of a conversation they have had so far.')
            for c in examples:
                if auto:
                    chat.add_user("Start the next conversation.")
                    chat.add_assistant(Message(
                        Paragraph(speaker1, "Dialogue", c[0]),
                        Paragraph(speaker1, "Action", c[1]),
                        Paragraph(speaker2, "Understanding", c[2]),
                        Paragraph(speaker2, "Thought", c[3]),
                        Paragraph(speaker2, "Dialogue", c[5]),
                        Paragraph(speaker2, "Action", c[4]),
                    ))
                elif rapid:
                    chat.add_user(c[0])
                    chat.add_assistant(c[5])
                else:
                    chat.add_user(Message(
                        Paragraph(speaker2, "Hearsay",
                                  f"{speaker1}は「{c[0]}」と言った。"),
                        Paragraph(speaker2, "Observation",
                                  f"{speaker1}は、{c[1]}"),
                    ))
                    chat.add_assistant(Message(
                        Paragraph(speaker2, "Understanding", c[2]),
                        Paragraph(speaker2, "Thought", c[3]),
                        Paragraph(speaker2, "Dialogue", c[5]),
                        Paragraph(speaker2, "Action", c[4]),
                    ))
            chat.add_system(
                f'However,{speaker1} and {speaker2} does not make exactly the same statements as the above sample.')

        chat.add_system(textwrap.dedent(f'''
        A summary of the conversation so far is as follows:
        {previous_summary}
        {last_summary}
        Here is what {speaker2} had in mind:
        {last_thought}
        ''').lstrip())

        if rapid:
            chat.add_system(
                f'You are {speaker2}. Now {speaker1} is going to talk to you, please reply as {speaker2}.')
        else:
            chat.add_system(
                f'This is where the rest of the conversation begins.')

        for c in conversations:
            if auto:
                chat.add_user("Start the next conversation.")
                chat.add_assistant(Message(
                    Paragraph(speaker1, "Dialogue", c[0]),
                    Paragraph(speaker1, "Action", c[1]),
                    Paragraph(speaker2, "Understanding", c[2]),
                    Paragraph(speaker2, "Thought", c[3]),
                    Paragraph(speaker2, "Dialogue", c[5]),
                    Paragraph(speaker2, "Action", c[4]),
                ))
            elif rapid:
                chat.add_user(c[0])
                chat.add_assistant(c[5])
            else:
                chat.add_user(Message(
                    Paragraph(speaker2, "Hearsay",
                              f"{speaker1}は「{c[0]}」と言った。"),
                    Paragraph(speaker2, "Observation", f"{speaker1}は、{c[1]}"),
                ))

                chat.add_assistant(Message(
                    Paragraph(speaker2, "Understanding", c[2]),
                    Paragraph(speaker2, "Thought", c[3]),
                    Paragraph(speaker2, "Dialogue", c[5]),
                    Paragraph(speaker2, "Action", c[4]),
                ))

        if auto:
            chat.add_user("Start the next conversation.")
            speaker1_speech = Paragraph(speaker1, "Dialogue", max_length=100)
            speaker1_action = Paragraph(speaker1, "Action", max_length=50)
        elif rapid:
            chat.add_user(speaker1_speech_text)
            speaker1_speech = Paragraph(
                speaker1, "Dialogue", speaker1_speech_text)
            speaker1_action = Paragraph(
                speaker1, "Action", speaker1_action_text)
        else:
            speaker1_speech = Paragraph(speaker2, "Hearsay")
            speaker1_action = Paragraph(speaker2, "Observation")

            if speaker1_speech_text == "……":
                speaker1_speech.text = f"何も聞こえなかった。"
            elif verbose:
                speaker1_speech.text = f"{speaker1}は「{speaker1_speech_text}」と、300文字くらいでの詳しい説明を求めてきた。"
            elif simple:
                speaker1_speech.text = f"{speaker1}は「{speaker1_speech_text}」という、質問をしてきた。"
            else:
                speaker1_speech.text = f"{speaker1}は「{speaker1_speech_text}」と言った。"
            speaker1_action.text = f"{speaker1}は、{speaker1_action_text}"

            if order:
                chat.add_user(Message(
                    speaker1_speech,
                    Paragraph(speaker2, "Observation", f"{speaker1}が指示した。"),
                ))
                chat.add_assistant(Message(
                    Paragraph(speaker2, "Understanding",
                              f"{speaker1}は自分に指示している。"),
                    Paragraph(speaker2, "Thought", f"{speaker1}の指示を実行しよう。"),
                ))
            elif simple:
                chat.add_user(Message(
                    speaker1_speech,
                    Paragraph(speaker2, "Observation",
                              f"{speaker1}が質問にシンプルに答えるように指示した。"),
                ))
                chat.add_assistant(Message(
                    Paragraph(speaker2, "Understanding",
                              f"{speaker1}は、短い回答を求めている。"),
                    Paragraph(speaker2, "Thought", f"シンプルに、10文字以内で手短かに答えよう。"),
                ))
            else:
                chat.add_user(Message(
                    speaker1_speech, speaker1_action
                ))

        result = chat.completion()

        speaker2_understand = Paragraph(
            speaker2, "Understanding", max_length=50)
        speaker2_thought = Paragraph(speaker2, "Thought", max_length=50)
        dialogue_length = 300 if verbose else 10 if simple else 100
        speaker2_speech = Paragraph(
            speaker2, "Dialogue", max_length=dialogue_length)
        speaker2_action = Paragraph(speaker2, "Action", max_length=50)

        if order or simple:
            result_message = Message(
                speaker2_speech, speaker2_action
            )
        elif auto:
            result_message = Message(
                speaker1_speech, speaker1_action, speaker2_understand, speaker2_thought, speaker2_speech, speaker2_action
            )
        elif rapid:
            pass
        else:
            result_message = Message(
                speaker2_understand, speaker2_thought, speaker2_speech, speaker2_action
            )

        if (rapid and result) or result_message.fill(result):
            if order:
                speaker2_understand.text = f"{speaker1}が{speaker1_speech_text}　と言った。"
                speaker2_thought.text = f"実行しよう。"
            elif simple:
                speaker2_understand.text = f"{speaker1}が{speaker1_speech_text}　と聞いてきた。"
                speaker2_thought.text = f"シンプルに答えよう。"
            elif auto:
                speaker1_speech_text = speaker1_speech.text
                speaker1_action_text = speaker1_action.text
                suffix = f'（{speaker1_action_text}）' if settings.show_action else ""
                if not settings.format_json:
                    print(f'{speaker1}: {speaker1_speech_text}{suffix}')
            elif rapid:
                speaker2_understand.text = f"{speaker1}が{speaker1_speech_text}　と言った。"
                speaker2_thought.text = f"返事をしよう。"
                speaker2_speech.text = result
                speaker2_action.text = "返事した。"

            speaker1_speech = Paragraph(
                speaker1, "Dialogue", speaker1_speech_text)
            speaker1_action = Paragraph(
                speaker1, "Action", speaker1_action_text)

            if settings.format_json:
                print(
                    Message(
                        speaker1_speech, speaker1_action, speaker2_understand, speaker2_thought, speaker2_action, speaker2_speech
                    ).to_json()
                )
            else:
                prefix = "" if settings.run_once and not auto else f'{speaker2}: '
                suffix = f'（{speaker2_action.text}）' if settings.show_action and not rapid else ""
                print(f'{prefix}{speaker2_speech.text}{suffix}')

            conversations.append(
                (speaker1_speech.text, speaker1_action.text, speaker2_understand.text, speaker2_thought.text, speaker2_action.text, speaker2_speech.text))

            if settings.send_speech and speaker2_speech.text:
                try:
                    url = settings.send_url.format(speech=speaker2_speech.text)
                    response = requests.get(url)
                except:
                    pass

            with open(settings.prompt_path, 'w', encoding='utf-8') as f:
                json.dump(chat.messages, f, ensure_ascii=False, indent=4)

            if speaker2_thought.text:
                with open(settings.thought_path, 'w', encoding='utf-8') as f:
                    f.write(speaker2_understand.text)
                    f.write("\n")
                    f.write(speaker2_thought.text)
                    if not settings.show_action:
                        f.write("\n")
                        f.write(speaker2_action.text)

        elif settings.show_error:
            print(result_message.error, file=sys.stderr)

        char_count = sum(len(s) for lst in conversations for s in lst)

        if len(conversations) >= 6 or char_count > 1000 or chat.total_tokens > 4000:
            last_conversations = oldConversations[-1]
            current_conversations = {"summary": "",
                                     "thought": "",
                                     "conversations": conversations[:3]}
            current_conversations["summary"], current_conversations["thought"] = summarize(
                title, story, speaker1, speaker2, last_conversations, current_conversations)

            if current_conversations["summary"] and current_conversations["thought"]:
                oldConversations.append(current_conversations)
                conversations = conversations[3:]

        with open(settings.conversation_path, 'w', encoding='utf-8') as f:
            data = {
                'title': data['title'],
                'speaker1': data['speaker1'],
                'speaker2': data['speaker2'],
                'story': data['story'],
                'examples': data['examples'],
                'oldConversations': oldConversations,
                'conversations': conversations,
            }
            json.dump(data, f, ensure_ascii=False, indent=4)
        if settings.run_once:
            break


@dataclass
class Settings:
    conversation_path: str
    model: str
    initial_input: str
    run_once: bool
    format_json: str
    rapid_mode_threshold: int
    send_url: Optional[str]
    show_action: bool
    show_error: bool

    def __post_init__(self):
        self.send_speech = bool(self.send_url)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(self.conversation_path):
            self.conversation_path = os.path.join(
                current_dir, self.conversation_path)
        self.prompt_path = os.path.join(current_dir, "prompt.txt")
        self.thought_path = os.path.join(current_dir, "thought.txt")
        self.summary_path = os.path.join(current_dir, "summary.txt")


class Chat:
    __response = {}
    __messages = []
    __initial_prompt = None

    def __init__(self, initial_prompt=None, model="gpt-3.5-turbo"):
        self.__messages = []
        self.__model = model
        if initial_prompt is not None:
            self.__initial_prompt = initial_prompt
            self.add_system(initial_prompt)

    @property
    def messages(self):
        return self.__messages

    @property
    def initial_prompt(self):
        return self.__initial_prompt

    @property
    def response(self):
        return self.__response

    @property
    def total_tokens(self):
        return self.__response.get(
            "usage", {}).get("total_tokens", 0)

    def completion(self):
        if self.total_tokens > 4000:
            for i, d in enumerate(self.__messages):
                if d.get("role") == "user":
                    del self.__messages[i]
                    break

        result = ""
        try:
            self.__response = openai.ChatCompletion.create(
                model=self.__model,
                messages=self.__messages
            )
            result = self.__response['choices'][0]['message']['content']

        except openai.error.APIError as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}", file=sys.stderr)
            pass
        except openai.error.APIConnectionError as e:
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}", file=sys.stderr)
            pass
        except openai.error.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(
                f"OpenAI API request exceeded rate limit: {e}", file=sys.stderr)
            pass

        # print(self.total_tokens)

        self.add_assistant(result)
        return result

    def clear(self):
        self.__messages = []
        self.add_system(self.__initial_prompt)
        return self

    def add_user(self, content):
        self.__messages.append({"role": "user", "content": str(content)})
        return self

    def add_system(self, content):
        self.__messages.append({"role": "system", "content": str(content)})
        return self

    def add_assistant(self, content):
        self.__messages.append({"role": "assistant", "content": str(content)})
        return self


class Message:
    def __init__(self, *paragraphs):
        self.__paragraphs = list(paragraphs)

    @property
    def paragraphs(self):
        return self.__paragraphs

    @property
    def error(self):
        return self.__error

    def __str__(self):
        return "\n".join(str(paragraph) for paragraph in self.__paragraphs)

    def add(self, paragraph):
        self.__paragraphs.append(paragraph)
        return self

    def fill(self, text):
        pattern = ""
        text = text.strip()
        if text and '<' in text and text[-1] != '>':
            text += '>'
        for p in self.__paragraphs:
            pattern += re.escape(f"{p.subject}'s {p.kind}")+r'\<(.*?)\>.*?'
        m = re.search(pattern, text, re.DOTALL)
        if m:
            groups = m.groups()
            for i, group in enumerate(groups):
                self.__paragraphs[i].text = group
            return True
        else:
            self.__error = text
            return False

    def to_json(self):
        return json.dumps([{'subject': p.subject, 'kind': p.kind, 'text': p.text}
                           for p in self.__paragraphs], ensure_ascii=False, indent=4)


class Paragraph:
    def __init__(self, subject, kind, text=None, max_length=1000):
        self.__kind = kind
        self.__text = text
        self.__subject = subject
        self.__max_length = max_length

    @property
    def kind(self):
        return self.__kind

    @property
    def text(self):
        return self.__text

    @text.setter
    def text(self, value):
        self.__text = self.__extract_sentences(value)

    @property
    def subject(self):
        return self.__subject

    def __extract_sentences(self, text):
        result = ''
        count = 0
        text = text.strip()
        for sentence in text.split('。'):
            result += sentence + '。'
            count += len(sentence) + 1
            if count > self.__max_length:
                break
        if len(result) > len(text):
            result = text
        return result

    def __str__(self):
        return f"{self.subject}'s {self.kind}<{self.text}>"


def parse_arguments() -> Settings:
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        metavar="json_path",
                        help='path to conversations JSON file')
    parser.add_argument('--model', '-m',
                        metavar='model',
                        choices=['gpt-3.5-turbo', 'gpt-4'],
                        default='gpt-3.5-turbo',
                        help='name of the OpenAI model to use (default: %(default)s)')
    parser.add_argument('--input', '-i',
                        dest='initial_input',
                        metavar='text',
                        help='the user\'s initial input (default: None)')
    parser.add_argument('--run-once', '-r',
                        action='store_true',
                        help='execute the API and immediately exit the script')
    parser.add_argument('--format', '-f',
                        dest='format_json',
                        action='store_true',
                        help='output the chat response in JSON format')
    parser.add_argument('--rapid-mode-threshold',
                        metavar='length',
                        default=6, type=int,
                        help='maximum length of input for rapid mode (default: %(default)s)')
    parser.add_argument('--send-url',
                        metavar='url',
                        help='URL to send AI speech to (example: http://localhost:8080/t={speech})')
    parser.add_argument('--no-show-action',
                        action='store_true',
                        help='do not show AI action output on console')
    parser.add_argument('--no-show-error',
                        action='store_true',
                        help='do not show error message if AI response could not be generated')
    args = parser.parse_args()

    return Settings(
        conversation_path=args.path,
        model=args.model,
        initial_input=args.initial_input,
        run_once=args.run_once,
        format_json=args.format_json,
        rapid_mode_threshold=args.rapid_mode_threshold,
        send_url=args.send_url,
        show_action=not args.no_show_action,
        show_error=not args.no_show_error,
    )


def summarize(title, story, speaker1, speaker2, last_conversations, current_conversations):
    chat = Chat()
    chat.add_system(textwrap.dedent(f'''
    This is a summary of the story "{title}".
    The synopsis of this story is as follows:
    {story}

    Please now summarize the conversation between {speaker1} and {speaker2} in Japanese in the following format.
    {Paragraph("Conversation", "Summary", "Summarize the content of this conversation in about 100 characters.")}
    {Paragraph(speaker2, "Thought", f"Summarize {speaker2}'s thoughts in this conversation in about 100 characters.")}

    Proper nouns, quantities, etc. mentioned in the text should be included in the summary as much as possible.
    ''').lstrip())

    conversation_text = "Conversation begins.\n"
    for c in last_conversations["conversations"]:
        conversation_text += f'{speaker1}は「{c[0]}」と言い、{c[1]}\n'
        conversation_text += f'{speaker2}は、{c[3]}と考え、「{c[5]}」と言い、{c[4]}\n'
    conversation_text += "Summarize the conversation so far.\n"

    chat.add_user(conversation_text)

    chat.add_assistant(Message(
        Paragraph("Conversation", "Summary", last_conversations["summary"]),
        Paragraph(speaker2, "Thought", last_conversations["thought"])
    ))

    conversation_text = "Conversation begins.\n"
    for c in current_conversations["conversations"]:
        conversation_text += f'{speaker1}は「{c[0]}」と言い、{c[1]}\n'
        conversation_text += f'{speaker2}は、{c[3]}と考え、「{c[5]}」と言い、{c[4]}\n'
    conversation_text += "Summarize the conversation so far.\n"

    chat.add_user(conversation_text)

    result = chat.completion()

    if result:
        summary_list = Message(
            Paragraph("Conversation", "Summary", max_length=150),
            Paragraph(speaker2, "Thought", max_length=100),
        )

        if summary_list.fill(result):
            with open(settings.summary_path, 'w', encoding='utf-8') as f:
                f.write("前々回の会話：")
                f.write(last_conversations["summary"])
                f.write("\n前回の会話：")
                f.write(summary_list.paragraphs[0].text)
                f.write(f"\n前回の{speaker2}の考え：")
                f.write(summary_list.paragraphs[1].text)

            return [
                summary_list.paragraphs[0].text,
                summary_list.paragraphs[1].text]
        elif settings.show_error:
            print(summary_list.error, file=sys.stderr)

    return ["", ""]


if __name__ == '__main__':
    settings = parse_arguments()
    main()
