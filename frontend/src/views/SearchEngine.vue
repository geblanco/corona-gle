<template>
	<b-container fluid="lg" style="height: 100vh;">
		<b-row align-v="center" style="height: 100vh;">
			<b-col align-self="center">
				<!-- LOGO -->
				<b-row class="text-center">
					<b-col>
						<img src="../assets/logo.png" alt="logo">	
					</b-col>
				</b-row>

				<!-- SEARCHER -->
				<b-row class="text-center">
					<b-col>
						<b-form style="margin-left: auto; margin-right: auto">
							<b-row class="justify-content-center">
								<b-form-input
									id="input-1"
									type="text"
									required
									placeholder="Your question..."
									class="search_input"
									v-model="current_question"
									></b-form-input>
								<b-form-select class="search_alg" v-model="current_algorithm" :options="algorithms"></b-form-select>
								<b-button :disabled="!validate_question()" :class="{active: validate_question()}" variant="primary" class="search_btn" @click="search">Search</b-button>
							</b-row>
						</b-form>
					</b-col>
				</b-row>

				<!-- ABOUT -->
				<br>
				<br>
				<!--<transition name="fade_more_info">-->
				<b-row v-if="!is_search" class="text-center" id="more_info">
					<b-col></b-col>
					<b-col cols="6" class="text-center">
						All searches will be stored in order to find out what are the most relevant and improve the algorithm into the future.
						For this reason we use cookies to track your queries.
						
						<br>
						<br>
						
						<h2>Do you want to help this engine work better? There is too much to do!</h2>						
						We need experts in the field of <b>epistemology</b>, <b>genetics</b>, <b>doctors</b> who can help us create a <b>more efficient search algorithm</b>: to know what is relevant in the papers, how to know if we are on the right track, validate if our algorithms are going well.
						
						<br>
						<br>
						We need experts in the field of <b>computer science</b>, <b>data science</b>, <b>maths</b>, <b>statistics</b>, math, ... To improve the current algorithms.

						<br>
						<br>
						<a target="_new" href="https://www.kaggle.com/arturkiulian/covid-19-global-team-collaboration-join-slack/">Join us! Your work could be relevant!</a>


					</b-col>
					<b-col></b-col>
				</b-row>
				<!--</transition>-->

				<!-- CURRENT SEARCH -->
				<b-row class="text-center" v-if="is_search">
					<b-col>
						<b-table
							id="table_jobs"
							@row-clicked="goto_result"
                            small
                            :items="current_results" 
                                :fields="[
                                { 
	                                key: 'rank', 
	                                sortable: false,
	                            }, 
                                { 
	                                key: 'reference_id', 
	                                sortable: false,
	                            }, , 
	                            { 
	                                key: 'title', 
	                                sortable: false,
	                            },
	                            /*{ 
	                                key: 'release_date', 
	                                sortable: true,
	                                formatter(value, key, item){
	                                	return moment(item.release_date).format('DD/MM/YYYY');
	                                },
	                                sortByFormatted: true,
	                            },*/
                                { 
                                	key: 'show_details', 
                                	label: 'Show',
                                }, 
	                            { 
	                                key: 'view'
	                            },
	                            
	                        ]"

                            >
                                <template v-slot:cell(show_details)="row">
                                    <b-button size="sm" @click="row.toggleDetails" class="mr-2">
                                        {{ row.detailsShowing ? 'Hide' : 'Show'}} Details
                                    </b-button>
                                </template>

		                         <template v-slot:cell(view)="row">
		                            <b-button>View</b-button>
		                        </template>

		                        <template v-slot:row-details="row">
		                        	<div style="background-color: #f2f2f2">
		                        		<!-- INFO PAPER -->
		                        		<h4>Abstract</h4>
		                        		<span>{{ row.item.abstract }}</span>

		                        		<br>
		                        		<br>
		                        		<h4>Body</h4>
		                        		<span>{{ row.item.body }}</span>

		                        	</div>
		                        </template>

						</b-table>
					</b-col>
				</b-row>
			</b-col>
		</b-row>
	</b-container>
</template>

<script>
import {ENDPOINTS} from '@/scripts/endpoints'
import axios from 'axios';
import JQuery from 'jquery';
let $ = JQuery;
let moment = require('moment');

export default {
	name: 'SearchEngine',
	components: {
	},

	data(){
		return {
			// CURRENT
			current_question: null,
			current_algorithm: 'WORD2VEC',
			current_results: [],

			// LIST OF...
			algorithms: ['WORD2VEC'],

			// OTHERS
			is_search: false
		};
	},

	methods: {
		moment: moment,

		send_request(call_data){
            // endpoint, method, data_uri={}, data={}, on_response=function(){}, on_finish=function(){}, on_error=function(){}
            axios.defaults.withCredentials = false;
            let endpoint_with_data = call_data['endpoint'];
            if(call_data['data_uri'] != undefined)
                endpoint_with_data = endpoint_with_data + '?' + $.param(call_data['data_uri']);

            let headers = undefined;
            /*if(this.$cookie.get("token")){
                headers = {headers: { "Authorization": 'JWT ' + this.$cookie.get("token") }};
            }*/

            if(call_data['method'] == 'GET'){
                return axios.get(endpoint_with_data, headers)
                .then(response => {
                    if(call_data['on_response'] !== undefined)
                        call_data['on_response'](response);
                })
                .catch(error => {
                    if(call_data['on_error'] !== undefined)
                        call_data['on_error'](error);
                })
                .finally(() => {
                    if(call_data['on_finish'] !== undefined)
                        call_data['on_finish']();
                });
            }else if(call_data['method'] == 'POST'){
                return axios.post(endpoint_with_data, call_data['data'], headers)
                .then(response => {
                    if(call_data['on_response'] !== undefined)
                        call_data['on_response'](response);
                })
                .catch(error => {
                    if(call_data['on_error'] !== undefined)
                        call_data['on_error'](error);
                })
                .finally(() => {
                    if(call_data['on_finish'] !== undefined)
                        call_data['on_finish']();
                });
            }
            
        },

        validate_question(){
        	return (this.current_question != null && this.current_question.length > 10);
        },

        search(e){
        	if(!this.validate_question())
        		return;

			let self = this;
    		self.is_search = true;
        	this.send_request({endpoint: ENDPOINTS.SEARCH, method: 'POST', data: {
        		query: self.current_question
        	}, on_response: function(response){
        		console.log(response)
            	self.current_results = response['data']['documents'];

            	// All ok
        		self.is_search = true;
	        },
	        on_error: function(error){
	            console.log(error);
	        }});
        },

        goto_result(r){
        	console.log(r);
        }
	},

	mounted(){
		let self = this;
		// Get algorithms
		this.send_request({endpoint: ENDPOINTS.INIT, method: 'GET', on_response: function(response){
			self.algorithms = response['data']['algorithms'];
        },
        on_error: function(error){
			console.log(error)

        }});
	},
}
</script>

<style scoped>

.search_input{
	border-radius: 5px 0px 0px 5px;
	width: 50%; 
	float: left;
}

.search_alg{
	float: left;
	width: 20%;
	border-radius: 0;
}

.search_btn{
	float: left;
	border-radius: 0px 5px 5px 0px;
    border-color: #ccc !important;
	background-color: #ccc !important;
}

.search_btn.active{
	color: #fff !important;
    background-color: #0062cc !important;
    border-color: #005cbf !important;
}

.fade_more_info-enter-active, .fade_more_info-leave-active {
  transition: opacity .5s;
}
.fade_more_info-enter, .fade_more_info-leave-to /* .fade-leave-active below version 2.1.8 */ {
  opacity: 0;
}

</style>
