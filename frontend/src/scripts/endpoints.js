let BACKEND_URL = process.env.VUE_APP_BACKEND_URL || 'http://127.0.0.1:7575';

export const ENDPOINTS = {
	INIT: BACKEND_URL + "/init",
	SEARCH: BACKEND_URL + "/search"
};