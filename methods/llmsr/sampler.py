# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Class for sampling new program skeletons. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from llmsr import evaluator
from llmsr import buffer
from llmsr import config as config_lib
from llmsr import profile
import requests
import json
import http.client
import os

import asyncio
from openai import OpenAI, AsyncOpenAI
from huggingface_hub import InferenceClient, AsyncInferenceClient
from multiprocessing import Process, Queue



class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1 

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config

    
    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                print("Break 1")
                break

            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                if profiler._cur_best_program_score == 0:
                    print("Break 2")
                    break
            
            prompt = self._database.get_prompt()
            
            reset_time = time.time()

            num_concurent = 20
            queue = asyncio.Queue(maxsize=self._samples_per_prompt)
            schedule_queue = asyncio.Queue(maxsize=num_concurent)

            async def sample_program():
                await schedule_queue.put(1)
                completion = await self._llm.async_draw_single_sample(prompt.code,self.config)
                await schedule_queue.get()

                await queue.put(completion)


                # queue.put(completion)
                # return completion
                # self._global_sample_nums_plus_one()
            

            async def process_sampled_program():
                for count in range(self._samples_per_prompt):
                    sample = await queue.get()

                    sample_time = (time.time() - reset_time) / (count + 1)

                    self._global_sample_nums_plus_one()
                    cur_global_sample_nums = self._get_global_sample_nums()
                    chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                    chosen_evaluator.analyse(
                        sample,
                        prompt.island_id,
                        prompt.version_generated,
                        **kwargs,
                        global_sample_nums=cur_global_sample_nums,
                        sample_time=sample_time
                    )
                # queue.put(completion)
                # return completion
                # self._global_sample_nums_plus_one()
            
            loop = asyncio.new_event_loop()
            tasks = [
                loop.create_task(sample_program()) for _ in range(self._samples_per_prompt)
            ] + [loop.create_task(process_sampled_program())]
            
            loop.run_until_complete(asyncio.wait(tasks))
            loop.close()

            # loop.run_until_complete(asyncio.gather(*tasks))
            # count = 0
            # while True:
            #     sample = queue.get()  # Read from the queue and do nothing
                
            #     count += 1
            #     sample_time = (time.time() - reset_time) / count

            #     self._global_sample_nums_plus_one()
            #     cur_global_sample_nums = self._get_global_sample_nums()
            #     chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
            #     chosen_evaluator.analyse(
            #         sample,
            #         prompt.island_id,
            #         prompt.version_generated,
            #         **kwargs,
            #         global_sample_nums=cur_global_sample_nums,
            #         sample_time=sample_time
            #     )
            #     if count == self._samples_per_prompt:
            #         break






            # samples = self._llm.draw_samples(prompt.code,self.config)
            # # print(samples)
            # sample_time = (time.time() - reset_time) / self._samples_per_prompt

            # # This loop can be executed in parallel on remote evaluator machines.
            # for sample in samples:
            #     self._global_sample_nums_plus_one()
            #     cur_global_sample_nums = self._get_global_sample_nums()
            #     chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
            #     chosen_evaluator.analyse(
            #         sample,
            #         prompt.island_id,
            #         prompt.version_generated,
            #         **kwargs,
            #         global_sample_nums=cur_global_sample_nums,
            #         sample_time=sample_time
            #     )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1






def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # for gpt APIs
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        
        # for mixtral
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        
        return code
    
    return sample



class LocalLLM(LLM):
    def __init__(self, 
                 samples_per_prompt: int, 
                 local_llm_url = None,
                 api_url = "api.openai.com",
                 api_key = None,
                 batch_inference: bool = True, 
                 trim=True) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        self._api_url = api_url
        self._api_key = api_key

        # port = os.environ.get("LLM_SR_SERVER_PORT", 10001)
        # url = f"http://127.0.0.1:{port}/completions"
        self._local_llm_url = local_llm_url
        instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. \
                             Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n")
        self._batch_inference = batch_inference
        # self._url = url
        self._instruction_prompt = instruction_prompt
        self._trim = trim

        if "openai" in self._api_url or "localhost" in self._api_url:
            # client = OpenAI(
            #     base_url=self._api_url,
            #     api_key=self._api_key,
            #     organization=os.environ["OPENAI_API_ORG"],
            # )
            self.client = AsyncOpenAI(
                base_url=self._api_url,
                api_key=self._api_key,
                # organization=os.environ["OPENAI_API_ORG"],
                timeout=60,
            )
        elif "api-inference.huggingface" in self._api_url:
            # client = InferenceClient(
            #     # base_url=self._api_url,
            #     api_key=self._api_key,
            #     headers={"x-use-cache": "false"}
            # )
            self.client = AsyncInferenceClient(
                # base_url=self._api_url,
                api_key=self._api_key,
                headers={"x-use-cache": "false"}
            )


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)

    async def async_draw_single_sample(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return await self._async_draw_single_sample_api(prompt, config)
        else:
            raise ValueError


    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
        raise NotImplementedError
        # instruction
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                # response from llm server
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        all_samples.append(response)

                # trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                return all_samples
            except Exception:
                continue


    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        prompt = '\n'.join([self._instruction_prompt, prompt])
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        async def req():
            while True:
                try:
                    completion = await self.client.chat.completions.create(
                                model=config.api_model, 
                                messages=messages, 
                                max_tokens=1024
                            )
                    response = completion.choices[0].message.content
                    break
                except:
                    continue
            return response
        
        loop = asyncio.new_event_loop()
        tasks = [
            loop.create_task(req()) for _ in range(self._samples_per_prompt)
        ]
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
        
        all_samples = []
        for task in tasks:
            response = task.result()
            if self._trim:
                response = _extract_body(response, config)
            all_samples.append(response)

        # for _ in range(self._samples_per_prompt):
        #     while True:
        #         try:
        #             messages = [
        #                 {
        #                     "role": "user",
        #                     "content": prompt
        #                 }
        #             ]
        #             # conn = http.client.HTTPSConnection(self._api_url)
        #             # payload = json.dumps({
        #             #     "max_tokens": 1024,
        #             #     "model": config.api_model,
        #             #     "messages": messages
        #             # })
        #             # headers = {
        #             #     'Authorization': f"Bearer {self._api_key}",
        #             #     'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        #             #     'Content-Type': 'application/json',
        #             #     "x-use-cache": "false"
        #             # }
        #             # conn.request("POST", "/v1/chat/completions", 
        #             #              payload, 
        #             #              headers)
        #             # res = conn.getresponse()
        #             # data = json.loads(res.read().decode("utf-8"))
        #             # response = data['choices'][0]['message']['content']

        #             completion = client.chat.completions.create(
        #                 model=config.api_model, 
        #                 messages=messages, 
        #                 max_tokens=1024
        #             )
        #             response = await completion.choices[0].message.content

        #             if self._trim:
        #                 response = _extract_body(response, config)
                    
        #             all_samples.append(response)
        #             break

        #         except Exception:
        #             continue
        
        return all_samples

    async def _async_draw_single_sample_api(self, prompt: str, config: config_lib.Config, timeout=None) -> Collection[str]:
        prompt = '\n'.join([self._instruction_prompt, prompt])
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        while True:
            try:
                if timeout is None:
                    completion = await self.client.chat.completions.create(
                                model=config.api_model, 
                                messages=messages, 
                                max_tokens=1024
                            )
                else:
                    completion = await self.client.chat.completions.create(
                                model=config.api_model, 
                                messages=messages, 
                                max_tokens=1024,
                                timeout=timeout
                            )
                response = completion.choices[0].message.content
                if self._trim:
                    response = _extract_body(response, config)
                break
            except:
                continue
        return response
        # return all_samples
        # return req()
    
    
    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': 1.0,
                'top_k': 50,
                'top_p': 1.0,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._local_llm_url, data=json.dumps(data), headers=headers)
        
        if response.status_code == 200: #Server status code 200 indicates successful HTTP request! 
            response = response.json()["content"]
            
            return response if self._batch_inference else response[0]




# class HuggingfaceLLM(LLM):
#     def __init__(self, 
#                  model="meta-llama/Llama-3.1-8B-Instruct",
#                  samples_per_prompt=4,
#                  trim=True):
#         api_key = os.environ['HF_INF_API_KEY']
#         self.client = InferenceClient(api_key=api_key,
#                                       headers={"x-use-cache": "false"})
#         self.model = model
#         self.samples_per_prompt = samples_per_prompt
#         self._trim = trim

#     def _chat_completion(self, mess_content, **kwargs):
#         messages = [
#             {
#                 "role": "user",
#                 "content": mess_content,
#             },
#         ]
#         completion = self.client.chat.completions.create(
#             model=self.model, 
#             messages=messages, 
#             **kwargs,
#         )
#         return completion.choices[0].message.content
    
#     def draw_samples(self, 
#                      prompt,
#                      config):
#         all_samples = []
#         for _ in range(self.samples_per_prompt):
#             response = self._chat_completion(prompt,
#                                              max_tokens=512,
#                                              temperature=0.8)
                    
#             if self._trim:
#                 response = _extract_body(response, config)
                    
#             all_samples.append(response)
#         return all_samples

import google.generativeai as genai
import concurrent.futures
# class GeminiLLM(LLM):
#     def __init__(self, 
#                  samples_per_prompt: int, 
#                  local_llm_url: str = None,  # Ignored for compatibility with LocalLLM
#                  api_url: str = None,  # Ignored for compatibility
#                  api_key: str = None,
#                  batch_inference: bool = True, 
#                  trim: bool = True) -> None:
#         """
#         Initialize the GeminiLLM class for generating equation program skeleton hypotheses using Gemini API.

#         Args:
#             samples_per_prompt: Number of samples to generate per prompt.
#             local_llm_url: Ignored (for compatibility with LocalLLM).
#             api_url: Ignored (for compatibility with LocalLLM).
#             api_key: API key for Gemini API (defaults to environment variable Google_API_KEY).
#             batch_inference: Use batch inference for generating multiple samples.
#             trim: Whether to trim the equation program skeleton body from samples.
#         """
#         super().__init__(samples_per_prompt)
#         self._api_key = api_key or os.getenv("Google_API_KEY")
#         if not self._api_key:
#             raise ValueError("Google_API_KEY must be set in environment or provided as api_key")
#         genai.configure(api_key=self._api_key)
#         self._model = genai.GenerativeModel('gemini-1.5-flash')  # Updated to latest model
#         self._batch_inference = batch_inference
#         self._trim = trim
#         self._instruction_prompt = ("You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
#                                   "Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n")

#     def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
#         """Generates multiple equation program skeleton hypotheses for the given prompt."""
#         prompt = '\n'.join([self._instruction_prompt, prompt])
#         all_samples = []

#         if self._batch_inference:
#             response = self._do_request(prompt, config)
#             all_samples.extend(response if isinstance(response, list) else [response])
#         else:
#             for _ in range(self._samples_per_prompt):
#                 response = self._do_request(prompt, config)
#                 all_samples.append(response)

#         if self._trim:
#             all_samples = [_extract_body(sample, config) for sample in all_samples]
#         return all_samples

#     async def async_draw_single_sample(self, prompt: str, config: config_lib.Config, timeout: float = None) -> str:
#         """Asynchronously generates a single equation program skeleton hypothesis."""
#         loop = asyncio.get_running_loop()
#         with concurrent.futures.ThreadPoolExecutor() as pool:
#             response = await loop.run_in_executor(pool, self._do_request, '\n'.join([self._instruction_prompt, prompt]), config)
#         if self._trim:
#             response = _extract_body(response, config)
#         return response

#     def _do_request(self, content: str, config: config_lib.Config) -> str:
#         """Synchronous request to Gemini API."""
#         content = content.strip('\n').strip()
#         for _ in range(10):  # Retry logic
#             try:
#                 response = self._model.generate_content(
#                     content,
#                     generation_config={
#                         'max_output_tokens': getattr(config, 'max_tokens', 1024),
#                         'temperature': getattr(config, 'temperature', 2.0)
#                     }
#                 )
#                 return response.text
#             except Exception as e:
#                 print(f"Attempt failed with error: {e}")
#                 time.sleep(2)
#         raise Exception("Failed to generate response after 10 attempts")
import logging
logging.basicConfig(
    filename='gemini_llm_responses.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class GeminiLLM(LLM):
    def __init__(self, 
                 samples_per_prompt: int, 
                 local_llm_url: str = None,  # Ignored for compatibility with LocalLLM
                 api_url: str = None,  # Ignored for compatibility
                 api_key: str = None,
                 batch_inference: bool = True, 
                 trim: bool = True) -> None:
        """
        Initialize the GeminiLLM class for generating equation program skeleton hypotheses using Gemini API.

        Args:
            samples_per_prompt: Number of samples to generate per prompt.
            local_llm_url: Ignored (for compatibility with LocalLLM).
            api_url: Ignored (for compatibility with LocalLLM).
            api_key: API key for Gemini API (defaults to environment variable Google_API_KEY).
            batch_inference: Use batch inference for generating multiple samples.
            trim: Whether to trim responses.
        """
        super().__init__(samples_per_prompt)
        self._api_key = api_key or os.getenv("Google_API_KEY")
        if not self._api_key:
            raise ValueError("Google_API_KEY must be set in environment or provided as api_key")
        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel('gemini-1.5-flash')  # Latest model
        self._batch_inference = batch_inference
        self._trim = trim
        self._sample_counter = 0  # Track sample order
        self._instruction_prompt = (
            "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
            "Generate a Python function 'equation' that computes the dipole moment based on the inputs. "
            "Return the response as a JSON object with a 'function' key containing the complete Python function code as a string. "
            "Consider the physical relationships between inputs and try diverse mathematical forms (e.g., polynomials, trigonometric, exponential). "
            "Example response:\n"
            "{\n"
            "  \"function\": \"def equation(Ef: np.ndarray, epsilon: np.ndarray, theta: np.ndarray, r: np.ndarray, params: np.ndarray) -> np.ndarray:\\n"
            "    \\\"\\\"\\\" Mathematical function for the dipole moment...\\\"\\\"\\\"\\n"
            "    output = params[0] * Ef * epsilon * (r**3) / np.sin(2*theta)\\n"
            "    return output\\n\"\n"
            "}\n\n"
            "Complete the 'equation' function below:\n"
        )

    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Generates multiple equation program skeleton hypotheses for the given prompt."""
        prompt = '\n'.join([self._instruction_prompt, prompt])
        all_samples = []

        if self._batch_inference:
            response = self._do_request(prompt, config)
            samples = self._parse_response(response, prompt)
            all_samples.extend(samples if isinstance(samples, list) else [samples])
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                futures = [pool.submit(self._do_request, prompt, config) for _ in range(self._samples_per_prompt)]
                for future in concurrent.futures.as_completed(futures):
                    response = future.result()
                    sample = self._parse_response(response, prompt)
                    all_samples.append(sample)

        if self._trim:
            all_samples = [_extract_body(sample, config) for sample in all_samples]
        return all_samples

    async def async_draw_single_sample(self, prompt: str, config: config_lib.Config, timeout: float = None) -> str:
        """Asynchronously generates a single equation program skeleton hypothesis."""
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            response = await loop.run_in_executor(pool, self._do_request, '\n'.join([self._instruction_prompt, prompt]), config)
        sample = self._parse_response(response, prompt)
        if self._trim:
            sample = _extract_body(sample, config)
        return sample

    def _do_request(self, content: str, config: config_lib.Config) -> str:
        """Synchronous request to Gemini API."""
        content = content.strip('\n').strip()
        for attempt in range(10):  # Retry logic
            try:
                response = self._model.generate_content(
                    content,
                    generation_config={
                        'max_output_tokens': getattr(config, 'max_tokens', 1024),
                        'temperature': getattr(config, 'temperature', 1.5)  # Increased for diversity
                    }
                )
                raw_response = response.text
                logging.info(f"Sample {self._sample_counter}: Raw response:\n{raw_response}")
                self._sample_counter += 1
                return raw_response
            except Exception as e:
                logging.error(f"Sample {self._sample_counter}: Attempt {attempt + 1} failed with error: {e}")
                time.sleep(2)
        raise Exception("Failed to generate response after 10 attempts")

    def _parse_response(self, response: str, prompt: str) -> str:
        """Parses the Gemini response to extract the equation function."""
        logging.info(f"Sample {self._sample_counter - 1}: Prompt:\n{prompt}")
        try:
            response_json = json.loads(response)
            function = response_json.get('function', response)
            logging.info(f"Sample {self._sample_counter - 1}: Parsed function:\n{function}")
            return function
        except json.JSONDecodeError:
            logging.warning(f"Sample {self._sample_counter - 1}: JSON parsing failed, using raw response")
            return response