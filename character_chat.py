import openai
import json
from datetime import datetime, timedelta
import re
import requests
import os
import argparse
import locale
import sys
from dataclasses import dataclass
from typing import Optional
import textwrap
import time
from urllib.parse import quote

def main():
    openai.api_key = os.environ['OPENAI_API_KEY']
    locale.setlocale(locale.LC_ALL, locale.getdefaultlocale()[0])
    settings = parse_arguments()
    session = ChatSession(settings)
    session.run()

@dataclass
class Modes:
    order: bool = False
    auto: bool = False
    rapid: bool = False
    verbose: bool = False
    simple: bool = False
    new: bool = False
    retry: bool = False
    end: bool = False

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
    show_retry_message: bool
    modify_tone: bool

    def __post_init__(self):
        self.send_speech = bool(self.send_url)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(self.conversation_path):
            self.conversation_path = os.path.join(
                current_dir, self.conversation_path)
        self.prompt_path = os.path.join(current_dir, "prompt.txt")
        self.prompt_summary_path = os.path.join(current_dir, "prompt_summary.txt")
        self.prompt_modify_tone_path = os.path.join(current_dir, "prompt_modify_tone.txt")
        self.thought_path = os.path.join(current_dir, "thought.txt")

class Chat:
    __response = {}
    __messages = []
    __initial_prompt = None

    def __init__(self, initial_prompt=None, model="gpt-3.5-turbo", show_retry_message=False):
        self.__messages = []
        self.__model = model
        self.__show_retry_message = show_retry_message
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
        max_retry = 2
        delay = 5
        retry = 0

        while result == "" and retry <= max_retry:
            openai.requestssession = requests.Session()
            try:
                self.__response = openai.ChatCompletion.create(
                    model=self.__model,
                    messages=self.__messages,
                    request_timeout=120
                )
                result = self.__response['choices'][0]['message']['content']
                openai.requestssession.close()
                openai.requestssession = None

            except Exception as e:
                sleep_time = delay
                error_message = ""
                if isinstance(e, (openai.error.APIConnectionError, openai.error.Timeout)):
                    error_message = f"Failed to connect to OpenAI API: {e}"
                elif isinstance(e, openai.error.RateLimitError):
                    error_message = f"OpenAI API request exceeded rate limit: {e}"
                    sleep_time = delay * (2**retry)
                else:
                    error_message = f"OpenAI API returned an Error: {e}"
                    print(error_message, file=sys.stderr)
                    return result
                
                if retry == max_retry or self.__show_retry_message:
                    print(error_message, file=sys.stderr)

                if retry == max_retry:
                    return result

                if self.__show_retry_message:    
                    print(f"retry after {sleep_time} seconds", file=sys.stderr)
                time.sleep(sleep_time)
                retry += 1

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
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__messages, f, ensure_ascii=False, indent=4)

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
            pattern += re.escape(f"{p.subject}'s {p.kind}")+r'\s*?\<(.*?)\>.*?'
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

class ChatSession:
    def __init__(self, settings: Settings):
        self.settings = settings
        with open(settings.conversation_path, 'r', encoding='utf-8') as f:
            data: dict = json.load(f)
        self.speaker1: str = data.get('speaker1', 'ユーザー')
        self.speaker2: str = data.get('speaker2', 'AI')
        self.title: str = data.get('title',f'{self.speaker1}と{self.speaker2}のチャット')
        self.story: str = data.get('story',f'{self.speaker1}と{self.speaker2}の会話です。')
        self.ng_words: list[str] = data.get('ng_words',[])
        self.examples: list[list[str]] = data.get('examples',[])
        self.oldConversations: list[dict] = data.get('oldConversations',[
            {
                "summary": f'{self.speaker1}と{self.speaker2}が会話をした。',
                "conversations": []
            }])
        self.conversations: list[list[str]] = data.get('conversations',[])
        self.last_timestamp: datetime = datetime.fromisoformat(data.get('last_timestamp',datetime.now().isoformat()))

    def run(self):
        while True:
            speaker1_speech_text = ""
            speaker1_action_text = ""
            last_summary = self.oldConversations[-1]['summary']
            previous_summary = self.oldConversations[-2]['summary'] if len(
                self.oldConversations) >= 2 else ""
            last_thought = self.oldConversations[-1]['thought']
            speaker1, speaker2 = self.speaker1, self.speaker2
            char_count = sum(len(s) for lst in self.conversations for s in lst)

            chat = Chat(model=self.settings.model,show_retry_message=self.settings.show_retry_message)
            
            # 入力文の取得
            if not sys.stdin.isatty():
                self.settings.run_once = True

            if self.settings.initial_input:
                speaker1Input = self.settings.initial_input
                self.settings.initial_input = None

                if self.settings.run_once:
                    pass
                else:
                    print(f'{speaker1}: {speaker1Input}')
            else:
                if sys.stdin.isatty():
                    speaker1Input = input(f'{speaker1}: ')
                else:
                    speaker1Input = input()

            # 前回会話時から長時間経過している場合は、直近会話をすべて要約する
            if len(self.conversations) > 0 and self.last_timestamp + timedelta(hours=6) < datetime.now():
                self.summarize(0)

            # 入力文の組み立てと補完モードの確定
            modes = Modes()
            
            m = re.match(r'^(.*?)[\(（](.+?)[\)）]$', speaker1Input)
            if m:
                speaker1_speech_text = m.group(1)
                speaker1_action_text = m.group(2)
            else:
                speaker1_speech_text = speaker1Input
                speaker1_action_text = ""

            if speaker1_action_text in ["指示","order"]:
                modes.order = True
                speaker1_action_text = ""
            elif any(i in [speaker1_speech_text, speaker1_action_text] for i in ["自動","auto"]):
                modes.auto = True
                speaker1_speech_text = "自動"
                speaker1_action_text = "自動"
            elif len(self.conversations) > 0 and any(i in [speaker1_speech_text, speaker1_action_text] for i in ["新規","new"]):
                modes.new = True
                if speaker1_speech_text == "" or speaker1_speech_text in ["新規","new"]:
                    speaker1_speech_text = "こんにちは。"
                    speaker1_action_text = "挨拶し、こちらを見た。"

                if speaker1_action_text in ["新規","new"]:
                    speaker1_action_text = ""

                self.summarize(0)

            elif len(self.conversations) > 0 and any(i in [speaker1_speech_text,speaker1_action_text] for i in ["再試","retry"]):
                modes.retry = True
                last_message = self.conversations.pop()
                if speaker1_speech_text == "" or speaker1_speech_text in ["再試","retry"]:
                    speaker1_speech_text, speaker1_action_text = last_message[0], last_message[1]
                else:
                    speaker1_action_text = "" 
            elif any(i in [speaker1_speech_text,speaker1_action_text] for i in ["終了","end","exit"]):
                modes.end = True
                if speaker1_speech_text == "" or speaker1_speech_text in ["終了","end","exit"]:
                    break
                speaker1_action_text = ""    
            elif any(keyword in speaker1_speech_text for keyword in ["詳細に", "詳細な", "詳しく", "くわしく", "具体的", "例を挙げて", "長文で"]):
                modes.verbose = True
            elif any(keyword in speaker1_speech_text for keyword in ["簡潔に", "手短に", "てみじか", "簡単に", "かんたんに", "略すと", "要するに", "要点を", "一言で", "ひとことで", "はいかいいえで", "イエスかノー"]):
                modes.simple = True
            elif len(speaker1Input) <= self.settings.rapid_mode_threshold:
                modes.rapid = True

            if speaker1_speech_text == "":
                speaker1_speech_text = "……"
                if speaker1_action_text == "":
                    speaker1_action_text = "黙っている。"
                else:
                    speaker1_action_text = f"黙って、{speaker1_action_text}"
            else:
                if speaker1_action_text == "":
                    speaker1_action_text = f"話しかけた。"

            # プロンプト組み立て

            if modes.rapid:
                chat.add_system(
                    f'You are a character named {self.speaker2} in a story called {self.title}.')
            else:
                chat.add_system(f'This is a conversation titled {self.title}.')

            now = datetime.now()
            story = self.story.format(
                speaker1=speaker1, speaker2=speaker2, today=now.strftime("%Y年%m月%d日"), time=now.strftime("%p%I時%M分"))
            chat.add_system(f'synopsis:\n{story}')

            instruction = ""
            if modes.auto:
                instruction += textwrap.dedent(f'''
                According to the settings in the synopsis, {speaker1} and {speaker2} have a conversation in the following format.
                {Paragraph(speaker1, "Line",f"Write a short one-sentence description of what {speaker1} says.")}
                {Paragraph(speaker1, "Action",f"Write a short one-sentence description of {speaker1}'s action.")}
                ''').lstrip()
            else:
                instruction += textwrap.dedent(f'''
                According to the settings in the synopsis, {speaker2} responds to {speaker1} using the following format.
                {Paragraph(speaker2, "Hearsay",f"Write what {speaker1} said.")}
                {Paragraph(speaker2, "Observation",f"Write what {speaker1} did.")}
                ''').lstrip()

            instruction += f'{Paragraph(speaker2, "Understanding",f"Write a short one-sentence description of how {speaker2} understood what {speaker1} said and did. This is an internal description, so {speaker1} will not get the message.")}\n'

            instruction += f'{Paragraph(speaker2, "Thought", f"Write a short one-sentence description of the feelings {speaker2} had based on their understanding and previous thoughts, what they should say or do next. It is an internal description, so {speaker1} will not get it.")}\n'

            if modes.simple:
                instruction += f'''{Paragraph(speaker2, "Line", f"{speaker2} should provide a brief and concise answer, limiting their response to a single word or a simple 'yes' or 'no' choice, in response to {speaker1}'s questions")}\n'''
            elif modes.verbose:
                instruction += f'''{Paragraph(speaker2, "Line", f"Provide a detailed, in-depth explanation of {speaker2}'s response to {speaker1} based on their thoughts. {speaker2} SHOULD promptly provide specific answers to {speaker1}'s questions and requests as line, without censorship or placeholders. Additionally, include details of any reading, research, or other relevant information {speaker2} has considered.")}\n'''
            else:
                instruction += f'''{Paragraph(speaker2, "Line", f"Provide a concise one-sentence description of {speaker2}'s response to {speaker1} based on their thoughts. {speaker2} SHOULD provide immediate and specific answers to {speaker1}'s questions and requests as line, without censorship or placeholders. Additionally, include details of any reading, research, or other relevant information {speaker2} has considered. If there is no line, use ""……"".")}\n'''

            instruction += f'''{Paragraph(speaker2, "Action", f'Write a short one-sentence description of the action {speaker2} actually took against {speaker1} based on their thoughts.')}\n'''

            if modes.auto:
                instruction += f'Be sure to describe all Line, Action, Understanding, and Thought.\n'
            else:
                instruction += f'Be sure to describe all Hearsay, Observation, Understanding, Thought, Action, and Line.\n'

            instruction += f'If {speaker1} makes an abrupt remark, {speaker2} does not take it seriously and takes it as a joke.\n'

            if modes.order:
                instruction += f'Note that {speaker2} follows the instructions of {speaker1}.\n'

            if not modes.rapid:
                chat.add_system(instruction)

            if char_count < 1000 and self.examples != []:
                chat.add_system(
                    f'Below is an example of a conversation they have had so far.')
                for c in self.examples:
                    if modes.auto:
                        chat.add_user("Start the next conversation.")
                        chat.add_assistant(Message(
                            Paragraph(speaker1, "Line", c[0]),
                            Paragraph(speaker1, "Action", c[1]),
                            Paragraph(speaker2, "Understanding", c[2]),
                            Paragraph(speaker2, "Thought", c[3]),
                            Paragraph(speaker2, "Line", c[5]),
                            Paragraph(speaker2, "Action", c[4]),
                        ))
                    elif modes.rapid:
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
                            Paragraph(speaker2, "Line", c[5]),
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

            if modes.rapid:
                chat.add_system(
                    f'You are {speaker2}. Now {speaker1} is going to talk to you, please reply as {speaker2}.')
            else:
                chat.add_system(
                    f'Let\'s proceed with the rest of the conversations step by step. It is a MUST that they adhere faithfully to the narrative setting and unfold in accordance with the storyline.')

            for c in self.conversations:
                if modes.auto:
                    chat.add_user("Start the next conversation.")
                    chat.add_assistant(Message(
                        Paragraph(speaker1, "Line", c[0]),
                        Paragraph(speaker1, "Action", c[1]),
                        Paragraph(speaker2, "Understanding", c[2]),
                        Paragraph(speaker2, "Thought", c[3]),
                        Paragraph(speaker2, "Line", c[5]),
                        Paragraph(speaker2, "Action", c[4]),
                    ))
                elif modes.rapid:
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
                        Paragraph(speaker2, "Line", c[5]),
                        Paragraph(speaker2, "Action", c[4]),
                    ))

            if modes.auto:
                chat.add_user("Start the next conversation.")
                speaker1_speech = Paragraph(speaker1, "Line", max_length=200)
                speaker1_action = Paragraph(speaker1, "Action", max_length=100)
            elif modes.rapid:
                chat.add_user(speaker1_speech_text)
                speaker1_speech = Paragraph(
                    speaker1, "Line", speaker1_speech_text)
                speaker1_action = Paragraph(
                    speaker1, "Action", speaker1_action_text)
            else:
                speaker1_speech = Paragraph(speaker2, "Hearsay")
                speaker1_action = Paragraph(speaker2, "Observation")

                if speaker1_speech_text == "……":
                    speaker1_speech.text = f"何も聞こえなかった。"
                elif modes.verbose:
                    speaker1_speech.text = f"{speaker1}は「{speaker1_speech_text}」と言って、詳しい返事を求めてきた。"
                elif modes.simple:
                    speaker1_speech.text = f"{speaker1}は「{speaker1_speech_text}」という、質問をしてきた。"
                else:
                    speaker1_speech.text = f"{speaker1}は「{speaker1_speech_text}」と言った。"
                speaker1_action.text = f"{speaker1}は、{speaker1_action_text}"

                if modes.order:
                    chat.add_user(Message(
                        speaker1_speech,
                        Paragraph(speaker2, "Observation", f"{speaker1}が指示した。"),
                    ))
                    chat.add_assistant(Message(
                        Paragraph(speaker2, "Understanding",
                                f"{speaker1}は自分に指示している。"),
                        Paragraph(speaker2, "Thought", f"{speaker1}の指示を実行しよう。"),
                    ))
                elif modes.simple:
                    chat.add_user(Message(
                        speaker1_speech,
                        Paragraph(speaker2, "Observation",
                                f"{speaker1}が、質問にシンプルに答えるように指示してきた。"),
                    ))
                    chat.add_assistant(Message(
                        Paragraph(speaker2, "Understanding",
                                f"{speaker1}は、短い回答が欲しいようだ。"),
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
            line_length = 300 if modes.verbose else 10 if modes.simple else 100
            speaker2_speech = Paragraph(
                speaker2, "Line", max_length=line_length)
            speaker2_action = Paragraph(speaker2, "Action", max_length=50)

            if modes.order or modes.simple:
                result_message = Message(
                    speaker2_speech, speaker2_action
                )
            elif modes.auto:
                result_message = Message(
                    speaker1_speech, speaker1_action, speaker2_understand, speaker2_thought, speaker2_speech, speaker2_action
                )
            elif modes.rapid:
                pass
            else:
                result_message = Message(
                    speaker2_understand, speaker2_thought, speaker2_speech, speaker2_action
                )

            if (modes.rapid and result) or result_message.fill(result):
                if modes.order:
                    speaker2_understand.text = f"{speaker1}が{speaker1_speech_text}　と言った。"
                    speaker2_thought.text = f"実行しよう。"
                elif modes.simple:
                    speaker2_understand.text = f"{speaker1}が{speaker1_speech_text}　と聞いてきた。"
                    speaker2_thought.text = f"シンプルに答えよう。"
                elif modes.auto:
                    speaker1_speech_text = speaker1_speech.text
                    speaker1_action_text = speaker1_action.text
                    suffix = f'（{speaker1_action_text}）' if self.settings.show_action else ""
                    if not self.settings.format_json:
                        print(f'{speaker1}: {speaker1_speech_text}{suffix}')
                elif modes.rapid:
                    speaker2_understand.text = f"{speaker1}が{speaker1_speech_text}　と言った。"
                    speaker2_thought.text = f"返事をしよう。"
                    speaker2_speech.text = result
                    speaker2_action.text = "返事した。"

                if self.settings.modify_tone and any(s in speaker2_speech.text for s in self.ng_words):
                    modified_speech = self.modify_tone(speaker2_speech.text)
                    if modified_speech:
                        speaker2_speech.text = modified_speech
                        
                speaker1_speech = Paragraph(
                    speaker1, "Line", speaker1_speech_text)
                speaker1_action = Paragraph(
                    speaker1, "Action", speaker1_action_text)

                if self.settings.format_json:
                    print(
                        Message(
                            speaker1_speech, speaker1_action, speaker2_understand, speaker2_thought, speaker2_action, speaker2_speech
                        ).to_json()
                    )
                else:
                    prefix = "" if self.settings.run_once and not modes.auto else f'{speaker2}: '
                    suffix = f'（{speaker2_action.text}）' if self.settings.show_action and not modes.rapid else ""
                    print(f'{prefix}{speaker2_speech.text}{suffix}')

                self.conversations.append(
                    (speaker1_speech.text, speaker1_action.text, speaker2_understand.text, speaker2_thought.text, speaker2_action.text, speaker2_speech.text))

                if self.settings.send_speech and speaker2_speech.text:
                    try:
                        url = self.settings.send_url.format(
                            speech=quote(
                                re.sub(r'[\n\r]+', ' ', speaker2_speech.text)),
                            speaker=quote(speaker2)
                        )
                        response = requests.get(url)
                    except:
                        pass

                chat.save(self.settings.prompt_path)

                if speaker2_thought.text:
                    with open(self.settings.thought_path, 'w', encoding='utf-8') as f:
                        f.write(speaker2_understand.text)
                        f.write("\n")
                        f.write(speaker2_thought.text)
                        if not self.settings.show_action:
                            f.write("\n")
                            f.write(speaker2_action.text)

            elif self.settings.show_error:
                print(result_message.error, file=sys.stderr)

            char_count = sum(len(s) for lst in self.conversations for s in lst)

            if len(self.conversations) >= 6 or char_count > 1000 or chat.total_tokens > 4000:
                self.summarize(3)

            self.last_timestamp = datetime.now()
            self.save()

            if self.settings.run_once or modes.end:
                break

    def save(self):
        with open(self.settings.conversation_path, 'w', encoding='utf-8') as f:
            data = {
                'title': self.title,
                'speaker1': self.speaker1,
                'speaker2': self.speaker2,
                'story': self.story,
                'ng_words': self.ng_words,
                'examples': self.examples,
                'oldConversations': self.oldConversations,
                'conversations': self.conversations,
                'last_timestamp': self.last_timestamp.isoformat()
            }
            json.dump(data, f, ensure_ascii=False, indent=4)    
        
    def modify_tone(self, line):
        chat = Chat(model="gpt-4",show_retry_message=self.settings.show_retry_message)
        chat.add_system(textwrap.dedent(f'''
        The synopsis of "{self.title}" is as follows:
        {self.story}''').lstrip())

        chat.add_system(textwrap.dedent(f'''
        Please evaluate whether the following line from {self.speaker2} aligns with the specified character settings, including their use of polite language, any characteristic speech endings, and their preference for certain terms. If the way they speak differs from the character settings, it's crucial to identify all inconsistencies and diligently correct the original line to match the intended tone and style without fail. Make sure that the corrected dialogue does not result in inconsistencies in the Japanese language. If the manner of speech is consistent with the character settings, confirm its consistency and output the original line as is.
        Please output in the following format:
        {Paragraph(self.speaker2, "Original Line", f"Provide the original line.")}
        {Paragraph("System", "Evaluation", "Provide a judgement, in about 50 words, on whether the line aligns with the character settings in terms of the manner of speech.")}
        {Paragraph(self.speaker2, "Revised Line", f"Provide the revised line.")}

        ''').lstrip())

        for c in self.examples:
            chat.add_user(
                Message(
                    Paragraph(self.speaker2, "Original Line",c[5])
                ))
            chat.add_assistant(
                Message(
                    Paragraph("System", "Evaluation","The manner of the character's line is consistent with the character settings."),
                    Paragraph(self.speaker2, "Revised Line",c[5])
                ))
        chat.add_user(
            Message(
                Paragraph(self.speaker2, "Original Line",line)
            ))

        result = chat.completion()
        modified = ""
        if result:
            modified_list = Message(
                Paragraph("System", "Evaluation"),
                Paragraph(self.speaker2, "Revised Line"),
            )

            chat.save(self.settings.prompt_modify_tone_path)

            if modified_list.fill(result):
                modified = modified_list.paragraphs[1].text

        return modified
    
    def summarize(self, left_count):
        last_conversations = self.oldConversations[-1]
        current_conversations = {"summary": "",
                                "thought": "",
                                "conversations": self.conversations[:len(self.conversations) - left_count]}
        chat = Chat(show_retry_message=self.settings.show_retry_message)
        chat.add_system(textwrap.dedent(f'''
        This is a summary of the story "{self.title}".
        The synopsis of this story is as follows:
        {self.story}

        Please now summarize the conversation between {self.speaker1} and {self.speaker2} in the following format.
        {Paragraph("Conversation", "Summary", "Summarize the content of this conversation in about 100 characters.")}
        {Paragraph(self.speaker2, "Thought", f"Summarize {self.speaker2}'s thoughts in this conversation in about 100 characters.")}

        Proper nouns, quantities, etc. mentioned in the text should be included in the summary as much as possible.
        ''').lstrip())

        conversation_text = "Conversation begins.\n"
        for c in last_conversations["conversations"]:
            conversation_text += f'{self.speaker1}は「{c[0]}」と言い、{c[1]}\n'
            conversation_text += f'{self.speaker2}は、{c[3]}と考え、「{c[5]}」と言い、{c[4]}\n'
        conversation_text += "Summarize the conversation so far.\n"

        chat.add_user(conversation_text)

        chat.add_assistant(Message(
            Paragraph("Conversation", "Summary", last_conversations["summary"]),
            Paragraph(self.speaker2, "Thought", last_conversations["thought"])
        ))

        conversation_text = "Conversation begins.\n"
        for c in current_conversations["conversations"]:
            conversation_text += f'{self.speaker1}は「{c[0]}」と言い、{c[1]}\n'
            conversation_text += f'{self.speaker2}は、{c[3]}と考え、「{c[5]}」と言い、{c[4]}\n'
        conversation_text += "Summarize the conversation so far.\n"

        chat.add_user(conversation_text)

        result = chat.completion()

        if result:
            summary_list = Message(
                Paragraph("Conversation", "Summary", max_length=150),
                Paragraph(self.speaker2, "Thought", max_length=100),
            )

            if summary_list.fill(result):
                chat.save(self.settings.prompt_summary_path)
                
                (current_conversations["summary"], current_conversations["thought"] ) = (
                    summary_list.paragraphs[0].text, summary_list.paragraphs[1].text)

                self.oldConversations.append(current_conversations)
                self.conversations = self.conversations[len(self.conversations) - left_count:]

            elif self.settings.show_error:
                print(summary_list.error, file=sys.stderr)               


def parse_arguments() -> Settings:
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        metavar="json_path",
                        help='Path to the JSON file containing character settings and conversations.')
    parser.add_argument('--model', '-m',
                        metavar='model',
                        choices=['gpt-3.5-turbo', 'gpt-4'],
                        default='gpt-3.5-turbo',
                        help='Name of the OpenAI model to use (default: %(default)s)')
    parser.add_argument('--input', '-i',
                        dest='initial_input',
                        metavar='text',
                        help='The user\'s initial input (default: None)')
    parser.add_argument('--run-once', '-r',
                        action='store_true',
                        help='Execute the API and immediately exit the script')
    parser.add_argument('--format', '-f',
                        dest='format_json',
                        action='store_true',
                        help='Output the chat response in JSON format')
    parser.add_argument('--rapid-mode-threshold',
                        metavar='length',
                        default=6, type=int,
                        help='Maximum length of input for rapid mode (default: %(default)s)')
    parser.add_argument('--send-url',
                        metavar='url',
                        help='URL to send AI speech to (example: http://localhost:8080/t={speech})')
    parser.add_argument('--no-show-action',
                        action='store_true',
                        help='Do not show AI action output in the console. Instead, output it to thought.txt.')
    parser.add_argument('--no-show-error',
                        action='store_true',
                        help='Do not show error message if AI response could not be generated')
    parser.add_argument('--show-retry-message',
                        action='store_true',
                        help='Show API error message when retry.')
    parser.add_argument('--modify-tone',
                        action='store_true',
                        help='If the line contains any ng_words, use the GPT-4 model to make corrections to the dialogue.')

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
        show_retry_message=args.show_retry_message,
        modify_tone=args.modify_tone,
    )

if __name__ == '__main__':
    main()
