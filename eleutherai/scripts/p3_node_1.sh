for TASK in adversarial_qa_dbidaf_generate_question amazon_polarity_Is_this_product_review_positive amazon_polarity_convey_negative_or_positive_sentiment amazon_polarity_would_you_buy app_reviews_generate_review cnn_dailymail_3.0.0_news_summary cnn_dailymail_3.0.0_write_an_outline common_gen_Put_together common_gen_topic_to_sentence cos_e_v1.11_description_question_option_text cos_e_v1.11_question_description_option_id cos_e_v1.11_rationale cosmos_qa_context_description_question_text cosmos_qa_description_context_question_answer_id cosmos_qa_no_prompt_text dbpedia_14_pick_one_category_for_the_following_text dream_generate_last_utterance duorc_ParaphraseRC_decide_worth_it duorc_ParaphraseRC_movie_director duorc_SelfRC_build_story_around_qa duorc_SelfRC_generate_question_by_answer gigaword_TLDR gigaword_make_a_title gigaword_write_its_sentence glue_mrpc_paraphrase glue_qqp_answer glue_qqp_quora imdb_Negation_template_for_positive_and_negative imdb_Reviewer_Opinion_bad_good_choices imdb_Writer_Expressed_Sentiment multi_news_summary_scenario paws_labeled_final_Concatenation_no_label paws_labeled_final_PAWS_ANLI_GPT3_no_label paws_labeled_final_context_question_no_label qasc_is_correct_2 qasc_qa_with_separated_facts_3 quail_context_description_question_answer_text quail_context_question_description_answer_id quail_description_context_question_answer_text quarel_choose_between quarel_testing_students quartz_having_read_above_passage quartz_use_info_from_question_paragraph quoref_Context_Contains_Answer quoref_Guess_Answer ropes_background_new_situation_answer ropes_plain_background_situation ropes_prompt_bottom_hint_beginning rotten_tomatoes_Movie_Expressed_Sentiment rotten_tomatoes_Reviewer_Expressed_Sentiment rotten_tomatoes_Text_Expressed_Sentiment samsum_Sum_up_the_following_dialogue samsum_Write_a_dialogue_that_match_this_summary sciq_Multiple_Choice_Closed_Book_ social_i_qa_Generate_the_question_from_the_answer trec_fine_grained_ABBR trec_fine_grained_ENTY trec_fine_grained_LOC_context_first trec_fine_grained_open_context_first trec_what_category_best_describe wiki_bio_key_content wiki_hop_original_choose_best_object_affirmative_2 wiki_hop_original_explain_relation wiki_qa_Decide_good_answer wiki_qa_Jeopardy_style wiki_qa_automatic_system wiqa_effect_with_label_answer wiqa_what_might_be_the_first_step_of_the_process xsum_DOC_given_above_write_one_sentence xsum_article_DOC_summary xsum_summarize_this_DOC_summary yelp_review_full_format_star; do
    PYTHONPATH=/home/connor/code/FLAN/:${PYTHONPATH} \
        python3 -m seqio.scripts.cache_tasks_main \
        --tasks=${TASK} \
        --output_cache_dir=gs://neo-datasets/zphang/multitask/t0_seqio/v1 \
        --module_import=promptsource.seqio_tasks \
        --alsologtostderr
done