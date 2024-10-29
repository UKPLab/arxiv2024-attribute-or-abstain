import copy
from typing import List, Any, Tuple

import numpy as np
import torch
import transformers
from intertext_graph import IntertextDocument
import rank_bm25
from scipy import sparse
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BatchEncoding

from attribution_eval.attribution_dataset import make_claim_from_evidence_inference, make_claim_from_contract_nli
from evaluation.common import BasePrediction, BaseInstance, CustomDataset
from evaluation.tasks.evidence_inference_task import EvidenceInferenceInstance
from evaluation.util import find_label, should_prediction_be_considered_for_attribution

TASK_TYPE_MAPPING = {
    'qasper': 'qa',
    'natural_questions': 'qa',
    'evidence_inference': 'ei',
    'govreport': 'sum',
    'contract_nli': 'nli',
    'wice': 'nli'
}


class BaseRetriever:
    def __init__(
            self,
            type: str, # post_hoc or retrieve_and_reduce,
            model_name: str,
            answer_has_multiple_statements: bool,
            k: int,
            threshold: float = None,
            classes: List[str] = None
    ):
        self.type = type
        self.model_name = model_name
        self.answer_has_multiple_statements = answer_has_multiple_statements
        self.k = k
        self.threshold = threshold
        self.classes = classes

    @staticmethod
    def load_model(
        type: str,
        model_name,
        answer_has_multiple_statements: bool,
        k: int,
        threshold: float = None,
        classes: List[str] = None,
        **kwargs
    ):
        if model_name in ['bm25', 'sbert', 'contriever']:
            return ClassicRetriever(
                type,
                model_name,
                answer_has_multiple_statements,
                k,
                threshold,
                classes=classes,
                **kwargs
            )
        else:
            raise NotImplementedError

    def post_hoc_retrieve_and_update_prediction(
            self,
            prediction: BasePrediction,
            instance: BaseInstance
    ) -> BasePrediction:
        if not should_prediction_be_considered_for_attribution(
            prediction,
            self.classes,
            prediction.task_name
        ):
            # Do not retrieve post hoc if answer does not need attribution (
            # e.g. because it is unanswerable)
            return prediction

        scores = self.collate_and_get_scores(instance, prediction)

        new_prediction = self.update_prediction_post_hoc(
            prediction,
            instance,
            scores
        )

        return new_prediction

    def collate_and_get_scores(
            self,
            instance: BaseInstance,
            prediction: BasePrediction = None
    ) -> List[float] | List[List[float]]:
        """Must be implemented by subclass
        Each returned list of scores should have the same length as the number
        of nodes in instance.document."""
        raise NotImplementedError

    def retrieve_and_reduce(
            self,
            instance: BaseInstance
    ) -> BaseInstance:
        """Get scores for nodes in document and return new instance in which
        the document is reduced to the nodes with top k scores."""
        scores = self.collate_and_get_scores(instance)

        new_instance = self.reduce_instance(instance, scores)

        return new_instance

    def update_prediction_post_hoc(
            self,
            prediction: BasePrediction,
            instance: BaseInstance,
            scores: List[float] | List[List[float]]
    ) -> BasePrediction:
        """Score the document nodes for (each statement in) predicted answer.
        Return new prediction with extraction nodes based on scores."""
        answer_has_multiple_statements = self.answer_has_multiple_statements

        all_nodes = instance.extraction_candidates
        extraction_nodes = []

        if answer_has_multiple_statements:
            for scores_list in scores:
                extraction_nodes.append(self._select_items_by_score(all_nodes, scores_list))
        else:
            extraction_nodes.extend(self._select_items_by_score(all_nodes, scores))

        new_prediction = BasePrediction(
            prediction.task_name,
            prediction.example_id,
            prediction.free_text_answer,
            extraction_nodes,
            prediction.raw_generation
        )

        return new_prediction

    def reduce_instance(
            self,
            instance: BaseInstance | EvidenceInferenceInstance,
            scores: List[float] | List[List[float]]
    ) -> BaseInstance | EvidenceInferenceInstance:
        """Remove all paragraphs from instance document that are not in
        top k"""
        # Get selection of nodes
        task_type = TASK_TYPE_MAPPING[instance.task_name]
        if task_type == 'sum':
            # The scores should be self similarity scores
            reduced_nodes = self._select_self_similarity(
                instance.extraction_candidates,
                scores
            )
        else:
            assert isinstance(scores[0], float)
            reduced_nodes = self._select_items_by_score(
                instance.extraction_candidates,
                scores
            )
        # Get ix of selected nodes
        reduced_node_idxs = [
            n.ix for n in reduced_nodes
        ]
        # Get ix of all extraction candidates
        extraction_candidate_idxs = [
            n.ix for n in instance.extraction_candidates
        ]

        # Make copy of original document
        new_doc: IntertextDocument = copy.deepcopy(instance.document)

        # Get all extraction nodes from new document
        if self.answer_has_multiple_statements:
            new_extraction_nodes = [
                [
                    [
                        new_doc.get_node_by_ix(n.ix)
                        for n in extraction_nodes_for_statement
                    ] for extraction_nodes_for_statement in annotation
                ] for annotation in instance.extraction_nodes
            ]
        else:
            new_extraction_nodes = [
                [
                    new_doc.get_node_by_ix(n.ix)
                    for n in annotation
                ] for annotation in instance.extraction_nodes
            ]

        # Remove nodes that are not in selection
        for node in new_doc.nodes:
            if node.ntype == 'p' and node.ix not in reduced_node_idxs:
                new_doc.remove_node(node, preserve_next_edge=True)

        # Get extraction candidates from new document as only those in the selection
        new_extraction_candidates = [
            new_doc.get_node_by_ix(ix)
            for ix in extraction_candidate_idxs
            if ix in reduced_node_idxs
        ]

        if isinstance(instance, EvidenceInferenceInstance):
            new_instance = EvidenceInferenceInstance(
                instance.task_name,
                instance.example_id,
                new_doc,
                instance.prompt,
                instance.question,
                instance.statement,
                instance.extraction_level,
                new_extraction_candidates,
                instance.free_text_answer,
                instance.answer_type,
                new_extraction_nodes,
                additional_info=instance.additional_info,
                answer_has_multiple_statements=instance.answer_has_multiple_statements,
                pmc_id=instance.pmc_id,
                prompt_id=instance.prompt_id,
                outcome=instance.outcome,
                intervention=instance.intervention,
                comparator=instance.comparator,
                labels=instance.labels,
                label_codes=instance.label_codes
            )
        elif isinstance(instance, BaseInstance):
            new_instance = BaseInstance(
                instance.task_name,
                instance.example_id,
                new_doc,
                instance.prompt,
                instance.question,
                instance.statement,
                instance.extraction_level,
                new_extraction_candidates,
                instance.free_text_answer,
                instance.answer_type,
                new_extraction_nodes,
                additional_info=instance.additional_info,
                answer_has_multiple_statements=instance.answer_has_multiple_statements
            )
        else:
            raise NotImplementedError

        return new_instance

    def _select_items_by_score(
            self,
            items: List[Any],
            scores: List[float]
    ) -> List[Any]:
        # Sort items according to scores in descending order
        # Weird behavior of ITG, so we first sort indices
        assert len(items) == len(scores)
        idcs = list(range(len(items)))
        sorted_idcs = [idx for _, idx in sorted(zip(scores, idcs), reverse=True)]
        sorted_items = [items[i] for i in sorted_idcs]
        sorted_scores = [scores[i] for i in sorted_idcs]

        selection = sorted_items[:self.k]

        if self.threshold:
            # Keep only items with scores larger than threshold
            selection = [
                item for score, item in zip(sorted_scores, selection)
                if score > self.threshold
            ]

        return selection

    def _select_self_similarity(
            self,
            items: List[Any],
            scores: List[List[float]]
    ) -> List[Any]:
        scores = np.array(scores)
        assert scores.shape[0] == scores.shape[1]
        self_sim_scores = np.sum(scores, axis=1)
        self_sim_scores = list(self_sim_scores)
        return self._select_items_by_score(items, self_sim_scores)


class ClassicRetriever(BaseRetriever):

    def __init__(
            self,
            type: str,
            model_name: str,
            answer_has_multiple_statements: bool,
            k: int,
            threshold: float = None,
            classes: List[str] = None,
            instances: List[BaseInstance] | CustomDataset = None,
            sbert_model_name: str = None,
            **kwargs
    ):
        super(ClassicRetriever, self).__init__(
            type,
            model_name,
            answer_has_multiple_statements,
            k,
            threshold,
            classes=classes
        )

        self.sbert = None
        self.sbert_query_encoder = None
        self.sbert_candidate_encoder = None
        if self.model_name == 'sbert':
            self.sbert = self.load_sbert(sbert_model_name)
            if isinstance(self.sbert, tuple):
                self.sbert_query_encoder, self.sbert_candidate_encoder = self.sbert
        self.contriever = None
        if self.model_name == 'contriever':
            self.tokenizer, self.contriever = self.load_contriever()

    @staticmethod
    def load_sbert(
            sbert_model_name
    ) -> SentenceTransformer | Tuple[SentenceTransformer, SentenceTransformer]:
        if 'dragon' in sbert_model_name:
            query_encoder = SentenceTransformer('nthakur/dragon-plus-query-encoder')
            candidate_encoder = SentenceTransformer('nthakur/dragon-plus-context-encoder')
            return query_encoder, candidate_encoder
        else:
            sbert = SentenceTransformer(sbert_model_name)
            return sbert

    @staticmethod
    def load_contriever():
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        model = transformers.AutoModel.from_pretrained("facebook/contriever-msmarco")
        if torch.cuda.is_available():
            model = model.cuda()
        return tokenizer, model


    def collate_and_get_scores(
            self,
            instance: BaseInstance,
            prediction: BasePrediction = None
    ) -> List[float] | List[List[float]]:
        task_type = TASK_TYPE_MAPPING[instance.task_name]

        if self.type == 'post_hoc':
            queries = self.make_post_hoc_queries(instance, prediction)
        else:
            queries = self.make_retrieve_and_reduce_queries(instance)

        candidates = [
            n.content
            for n in instance.extraction_candidates
        ]

        scores = []

        scores.extend(self.score(queries, candidates))

        if(
            (not self.answer_has_multiple_statements)
            or (self.type == 'retrieve_and_reduce' and task_type != 'sum')
        ):
            scores = scores[0]

        return scores

    def make_post_hoc_queries(
            self,
            instance: BaseInstance | EvidenceInferenceInstance,
            prediction: BasePrediction
    ) -> List[str]:

        def make_qa_query(question, answer) -> str:
            query = f'{question} {answer}'
            return query

        def make_sum_query(answer) -> str:
            return answer

        def make_nli_query(statement, answer) -> str:
            query = f'{statement}'
            return query

        def make_contract_nli_query(statement, answer) -> str:
            label = find_label(self.classes, answer)
            return make_claim_from_contract_nli(
                statement,
                label
            )

        def make_ei_query(
                outcome,
                comparator,
                intervention,
                answer
        ) -> str:
            label = find_label(self.classes, answer)
            query = make_claim_from_evidence_inference(
                label,
                outcome,
                comparator,
                intervention
            )
            return query

        task_type = TASK_TYPE_MAPPING[instance.task_name]

        queries = []
        if self.answer_has_multiple_statements:
            free_text_answers = prediction.free_text_answer
        else:
            free_text_answers = [prediction.free_text_answer]

        for ans in free_text_answers:
            if instance.task_name == 'contract_nli':
                # Special handling of contract nli
                queries.append(make_contract_nli_query(instance.statement, ans))
            elif task_type == 'qa':
                queries.append(make_qa_query(instance.question, ans))
            elif task_type == 'sum':
                queries.append(make_sum_query(ans))
            elif task_type == 'ei':
                queries.append(make_ei_query(
                    instance.outcome,
                    instance.comparator,
                    instance.intervention,
                    ans
                ))
            elif task_type == 'nli':
                queries.append(make_nli_query(instance.statement, ans))
            else:
                raise NotImplementedError

        return queries

    def make_retrieve_and_reduce_queries(
            self,
            instance: BaseInstance | EvidenceInferenceInstance
    ) -> List[str]:
        def make_qa_query(question):
            return question

        def make_nli_query(statement):
            return statement

        def make_sum_query(node_content):
            return node_content

        def make_ei_query(question):
            return question

        task_type = TASK_TYPE_MAPPING[instance.task_name]

        queries = []

        if task_type == 'qa':
            queries.append(make_qa_query(instance.question))
        elif task_type == 'ei':
            queries.append(make_ei_query(
                instance.question
            ))
        elif task_type == 'nli':
            queries.append(make_nli_query(instance.statement))
        elif task_type == 'sum':
            for n in instance.extraction_candidates:
                queries.append(make_sum_query(n.content))

        return queries

    def score(
            self,
            queries: List[str],
            candidates: List[str]
    ) -> List[List[float]]:
        if self.model_name == 'bm25':
            scores = self.score_bm25(queries, candidates)
        elif self.model_name == 'sbert':
            scores = self.score_sbert(queries, candidates)
        elif self.model_name == 'contriever':
            scores = self.score_contriever(queries, candidates)
        else:
            raise NotImplementedError
        return scores

    def score_bm25(
            self,
            queries: List[str],
            candidates: List[str]
    ) -> List[List[float]]:
        scores = []
        tokenized_candidates = [
            c.split(' ') for c in candidates
        ]
        bm25 = rank_bm25.BM25Okapi(tokenized_candidates)
        for query in queries:
            scores.append(list(bm25.get_scores(query.split(' '))))
        return list(scores)

    def score_sbert(
            self,
            queries: List[str],
            candidates: List[str]
    ) -> List[List[float]]:
        if self.sbert_query_encoder is not None:
            query_embeddings = self.sbert_query_encoder.encode(queries, convert_to_tensor=True)
            candidate_embeddings = self.sbert_candidate_encoder.encode(candidates, convert_to_tensor=True)
            tokenizer_name = self.sbert_query_encoder.tokenizer.name_or_path
        else:
            query_embeddings = self.sbert.encode(queries, convert_to_tensor=True)
            candidate_embeddings = self.sbert.encode(candidates, convert_to_tensor=True)
            tokenizer_name = self.sbert.tokenizer.name_or_path
        if tokenizer_name in [
            'sentence-transformers/gtr-t5-large',
            'nthakur/dragon-plus-query-encoder'
        ]:
            scores = query_embeddings @ candidate_embeddings.transpose(1, 0)
        else:
            scores = util.cos_sim(query_embeddings, candidate_embeddings)
        all_scores = []
        for row in scores:
            all_scores.append(row.tolist())

        return all_scores

    def make_batches(
            self,
            encoding: BatchEncoding,
            batch_size: int = 8
    ) -> List[BatchEncoding]:
        n = len(encoding['input_ids'])
        batches = []
        for i in range(0, n, batch_size):
            batch = {
                k: v[i:i+batch_size]
                for k, v in encoding.items()
            }
            batches.append(batch)
        return batches

    def score_contriever(
            self,
            queries: List[str],
            candidates: List[str]
    ):
        # Mean pooling
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
        queries_tokenized = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        candidates_tokenized = self.tokenizer(
            candidates,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        query_batches = self.make_batches(queries_tokenized)
        query_embeddings = []
        for batch in query_batches:
            input_ids = batch['input_ids'].to(self.contriever.device)
            attention_mask = batch['attention_mask'].to(self.contriever.device)
            with torch.no_grad():
                query_token_embeddings = (self.contriever(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )[0])
            query_embeddings.append(mean_pooling(
                query_token_embeddings,
                attention_mask
            ).to('cpu'))
        query_embeddings = torch.concat(query_embeddings)
        candidate_batches = self.make_batches(candidates_tokenized)
        candidate_embeddings = []
        for batch in candidate_batches:
            input_ids = batch['input_ids'].to(self.contriever.device)
            attention_mask = batch['attention_mask'].to(self.contriever.device)
            with torch.no_grad():
                candidate_token_embeddings = (self.contriever(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )[0])
            candidate_embeddings.append(mean_pooling(
                candidate_token_embeddings,
                attention_mask
            ).to('cpu'))
        candidate_embeddings = torch.concat(candidate_embeddings)
        scores = query_embeddings @ candidate_embeddings.transpose(1,0)

        all_scores = []
        for row in scores:
            all_scores.append(row.tolist())

        return all_scores

class BM25:
    """
    This implementation of BM25 allows scoring documents that were not in the
    corpus used to compute the term frequency statistics.
    We are not using it right now because we are computing the frequency
    statistics for each document individually, as each document might have
    specific terms that were not in the training corpus.
    from https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8"""
    def __init__(
            self,
            corpus: List[str]
    ):
        self.bm25_avdl = None
        self.bm25_b = 0.75
        self.bm25_k1 = 1.5

        self.vectorizer = TfidfVectorizer(
            norm=None,
            smooth_idf=False
        )
        self.vectorizer.fit(corpus)
        y = super(TfidfVectorizer, self.vectorizer).transform(corpus)
        self.bm25_avdl = y.sum(1).mean()

    def transform(self, query, candidates):
        b, k1, avdl = self.bm25_b, self.bm25_k1, self.bm25_avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(candidates)
        len_X = X.sum(1).A1
        query, = super(TfidfVectorizer, self.vectorizer).transform([query])
        assert sparse.isspmatrix_csr(query)

        # convert to csc for better column slicing
        X = X.tocsc()[:, query.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, query.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1

if __name__ == '__main__':
    """Testing: Scores are different between the two implementations,
    but the ratios among the scores from both methods are the same 
    when using lambda x: x.split(' ') as tokenizing function"""
    from rank_bm25 import BM25Okapi

    corpus = [
        'i like pizza very much',
        'llama3 just came out',
        'do worry be happy yes'
    ]
    tokenized_corpus = [
        s.split(' ') for s in corpus
    ]

    query = 'do you worry about pizza or just about llama3'
    tokenized_query = query.split(' ')

    r_bm25 = BM25Okapi(tokenized_corpus, epsilon=1)
    r_bm25_scores = r_bm25.get_scores(tokenized_query)

    self_bm25 = BM25(corpus)
    self_bm25_scores = self_bm25.transform(query, corpus)

    pass

