import numpy as np
import pandas as pd
import logging
import os
import json
import re
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib

logger = logging.getLogger("sentiment_analyzer")

class SentimentAnalyzer:
    """감정 분석 처리 엔진"""
    
    def __init__(self, use_pretrained=True, model_path='saved_models/sentiment'):
        """초기화
        
        Args:
            use_pretrained (bool): 사전 훈련된 모델 사용 여부
            model_path (str): 모델 저장 경로
        """
        self.model_path = model_path
        self.nltk_initialized = False
        
        # NLTK 데이터 초기화
        try:
            nltk.data.find('vader_lexicon')
            self.nltk_initialized = True
            self.vader = SentimentIntensityAnalyzer()
        except LookupError:
            try:
                nltk.download('vader_lexicon')
                self.vader = SentimentIntensityAnalyzer()
                self.nltk_initialized = True
            except Exception as e:
                logger.error(f"NLTK 초기화 실패: {e}")
        
        # 감정 분석 모델 로드
        self.model = None
        self.vectorizer = None
        
        if use_pretrained:
            self._load_model()
            
        # 암호화폐 용어 감정 사전
        self.crypto_lexicon = {
            # 긍정적 용어
            'bullish': 2.0,
            'bull': 1.5,
            'rally': 1.5,
            'uptrend': 1.5,
            'soar': 1.8,
            'surge': 1.7,
            'breakout': 1.5,
            'adoption': 1.2,
            'institutional': 1.0,
            'adoption': 1.2,
            'halving': 1.3,
            'hodl': 1.0,
            'mooning': 2.0,
            'to the moon': 2.0,
            
            # 부정적 용어
            'bearish': -2.0,
            'bear': -1.5,
            'crash': -2.0,
            'dump': -1.7,
            'dip': -1.0,
            'correction': -1.0,
            'ban': -1.5,
            'regulation': -1.0,
            'hack': -2.0,
            'scam': -2.0,
            'bubble': -1.5,
            'collapse': -2.0,
            'fud': -1.8,
            'fear': -1.5,
            'uncertainty': -1.0,
            'doubt': -1.0
        }
        
        logger.info("감정 분석 엔진 초기화 완료")
    
    def _load_model(self):
        """감정 분석 모델 로드"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            model_file = os.path.join(self.model_path, 'sentiment_model.joblib')
            vectorizer_file = os.path.join(self.model_path, 'tfidf_vectorizer.joblib')
            
            if os.path.exists(model_file) and os.path.exists(vectorizer_file):
                self.model = joblib.load(model_file)
                self.vectorizer = joblib.load(vectorizer_file)
                logger.info("사전 훈련된 감정 분석 모델 로드됨")
            else:
                logger.warning("사전 훈련된 모델 파일이 없습니다")
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {e}")
    
    def preprocess_text(self, text):
        """텍스트 전처리
        
        Args:
            text (str): 원본 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        if not text:
            return ""
            
        # 소문자 변환
        text = text.lower()
        
        # URL 제거
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # HTML 태그 제거
        text = re.sub(r'<.*?>', '', text)
        
        # 특수문자 제거
        text = re.sub(r'[^\w\s]', '', text)
        
        # 여러 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_text(self, text):
        """텍스트 감정 분석
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            dict: 감정 분석 결과
        """
        if not text:
            return {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
            
        # 텍스트 전처리
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
            
        # VADER 분석
        if self.nltk_initialized:
            vader_scores = self.vader.polarity_scores(processed_text)
            
            # 암호화폐 용어 감정 반영
            for term, score in self.crypto_lexicon.items():
                if term in processed_text:
                    # 복합 점수에 가중치 적용
                    multiplier = 1.0
                    if abs(score) >= 2.0:  # 매우 강한 감정
                        multiplier = 2.0
                    elif abs(score) >= 1.5:  # 강한 감정
                        multiplier = 1.5
                    
                    # 각 발견에 대해 복합 점수 조정
                    occurrences = len(re.findall(r'\b' + re.escape(term) + r'\b', processed_text))
                    adjustment = score * 0.1 * multiplier * occurrences
                    
                    # VADER 점수 조정
                    vader_scores['compound'] = max(min(vader_scores['compound'] + adjustment, 1.0), -1.0)
                    
                    # 긍정/부정 조정
                    if score > 0:
                        vader_scores['pos'] = min(vader_scores['pos'] + (adjustment/4), 1.0)
                        vader_scores['neg'] = max(vader_scores['neg'] - (adjustment/4), 0.0)
                    else:
                        vader_scores['neg'] = min(vader_scores['neg'] + (abs(adjustment)/4), 1.0)
                        vader_scores['pos'] = max(vader_scores['pos'] - (abs(adjustment)/4), 0.0)
                    
                    # 중립 조정
                    vader_scores['neu'] = 1 - (vader_scores['pos'] + vader_scores['neg'])
            
            return vader_scores
            
        # ML 모델 사용 (VADER 사용 불가 시)
        elif self.model and self.vectorizer:
            try:
                # 벡터화
                features = self.vectorizer.transform([processed_text])
                
                # 예측
                prediction = self.model.predict_proba(features)[0]
                
                # 결과 형식 변환 (VADER와 유사하게)
                pos_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else -1
                neg_idx = list(self.model.classes_).index(0) if 0 in self.model.classes_ else -1
                
                pos = prediction[pos_idx] if pos_idx >= 0 else 0
                neg = prediction[neg_idx] if neg_idx >= 0 else 0
                neu = 1 - (pos + neg)
                
                # 복합 점수 계산 (-1: 극단적 부정, 1: 극단적 긍정)
                compound = (pos - neg) * (1 - neu)
                
                return {
                    'pos': pos,
                    'neg': neg,
                    'neu': neu,
                    'compound': compound
                }
            except Exception as e:
                logger.error(f"ML 모델 기반 분석 오류: {e}")
                return {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
        
        # 모든 방법 실패 시 중립 반환
        return {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
    
    def analyze_batch(self, texts):
        """대량 텍스트 분석
        
        Args:
            texts (list): 분석할 텍스트 목록
            
        Returns:
            list: 감정 분석 결과 목록
        """
        results = []
        for text in texts:
            results.append(self.analyze_text(text))
        return results
    
    def extract_topics(self, texts, num_topics=5):
        """토픽 모델링으로 주요 이슈 추출
        
        Args:
            texts (list): 텍스트 목록
            num_topics (int): 추출할 토픽 수
            
        Returns:
            list: 토픽 목록
        """
        try:
            # 간단한 LDA 기반 토픽 모델링 (실제 구현 시 확장)
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            
            # 전처리
            processed_texts = [self.preprocess_text(text) for text in texts if text]
            
            if not processed_texts:
                return []
                
            # 벡터화
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = vectorizer.fit_transform(processed_texts)
            
            # LDA 모델
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(dtm)
            
            # 토픽별 주요 단어 추출
            words = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                # 상위 10개 단어
                top_words_idx = topic.argsort()[:-11:-1]
                top_words = [words[i] for i in top_words_idx]
                
                # 토픽 저장
                topics.append({
                    'id': topic_idx,
                    'words': top_words,
                    'weight': float(topic.sum() / lda.components_.sum())
                })
            
            return topics
            
        except Exception as e:
            logger.error(f"토픽 추출 중 오류: {e}")
            return []
    
    def calculate_sentiment_score(self, sentiment_data):
        """종합 감정 점수 계산
        
        Args:
            sentiment_data (dict): 감정 데이터
            
        Returns:
            dict: 종합 감정 점수
        """
        # 기본값 설정
        result = {
            'overall_score': 0,
            'news_score': 0,
            'social_score': 0,
            'market_score': 0,
            'components': {}
        }
        
        try:
            scores = []
            weights = []
            
            # 1. 뉴스 기사 감정
            if 'news' in sentiment_data and sentiment_data['news']:
                news_scores = []
                
                # CryptoPanic 뉴스
                if 'cryptopanic' in sentiment_data['news']:
                    cp_articles = sentiment_data['news']['cryptopanic']
                    if cp_articles:
                        for article in cp_articles:
                            if 'sentiment' in article and article['sentiment'] is not None:
                                # 이미 감정 점수가 있는 경우
                                news_scores.append(article['sentiment'] / 10)  # -1~1 범위로 정규화
                            else:
                                # 타이틀 분석
                                sentiment = self.analyze_text(article['title'])
                                news_scores.append(sentiment['compound'])
                
                # 다른 뉴스 소스
                if 'coindesk' in sentiment_data['news']:
                    cd_articles = sentiment_data['news']['coindesk']
                    if cd_articles:
                        for article in cd_articles:
                            sentiment = self.analyze_text(article['title'])
                            news_scores.append(sentiment['compound'])
                
                if news_scores:
                    avg_news_score = sum(news_scores) / len(news_scores)
                    result['news_score'] = avg_news_score
                    result['components']['news'] = {
                        'score': avg_news_score,
                        'count': len(news_scores),
                        'weight': 0.4
                    }
                    
                    scores.append(avg_news_score)
                    weights.append(0.4)  # 뉴스의 가중치
            
            # 2. 소셜 미디어 감정
            if 'social' in sentiment_data and sentiment_data['social']:
                social_scores = []
                
                # Reddit 감정
                if 'reddit' in sentiment_data['social'] and sentiment_data['social']['reddit']:
                    reddit_data = sentiment_data['social']['reddit']
                    if 'sentiment_score' in reddit_data:
                        social_scores.append(reddit_data['sentiment_score'])
                
                # Twitter 감정
                if 'twitter' in sentiment_data['social'] and sentiment_data['social']['twitter']:
                    twitter_data = sentiment_data['social']['twitter']
                    if 'sentiment_score' in twitter_data:
                        social_scores.append(twitter_data['sentiment_score'])
                
                if social_scores:
                    avg_social_score = sum(social_scores) / len(social_scores)
                    result['social_score'] = avg_social_score
                    result['components']['social'] = {
                        'score': avg_social_score,
                        'count': len(social_scores),
                        'weight': 0.35
                    }
                    
                    scores.append(avg_social_score)
                    weights.append(0.35)  # 소셜 미디어의 가중치
            
            # 3. 시장 지표
            if 'market_indicators' in sentiment_data and sentiment_data['market_indicators']:
                market_scores = []
                
                # 공포·탐욕 지수
                if 'fear_greed_index' in sentiment_data['market_indicators'] and sentiment_data['market_indicators']['fear_greed_index']:
                    fear_greed = sentiment_data['market_indicators']['fear_greed_index']
                    if 'value' in fear_greed:
                        # 0-100 범위를 -1~1 범위로 변환
                        fg_score = (fear_greed['value'] / 50) - 1
                        market_scores.append(fg_score)
                
                if market_scores:
                    avg_market_score = sum(market_scores) / len(market_scores)
                    result['market_score'] = avg_market_score
                    result['components']['market'] = {
                        'score': avg_market_score,
                        'count': len(market_scores),
                        'weight': 0.25
                    }
                    
                    scores.append(avg_market_score)
                    weights.append(0.25)  # 시장 지표의 가중치
            
            # 종합 점수 계산 (가중평균)
            if scores and weights:
                weighted_sum = sum(s * w for s, w in zip(scores, weights))
                total_weight = sum(weights)
                
                if total_weight > 0:
                    result['overall_score'] = weighted_sum / total_weight
                
            return result
            
        except Exception as e:
            logger.error(f"감정 점수 계산 중 오류: {e}")
            return result