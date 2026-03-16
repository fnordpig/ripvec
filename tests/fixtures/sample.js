function fetchData(url) {
    return fetch(url).then(r => r.json());
}

class DataService {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    getData(path) {
        return fetchData(this.baseUrl + path);
    }
}
