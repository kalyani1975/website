import React from "react";
import { HashRouter, Switch, Route } from "react-router-dom";

import Home from "./Containers/Home"
import BlogPost from "./Containers/BlogPost";

var createBrowserHistory = require("history").createBrowserHistory;
const history = createBrowserHistory();

const Router = () => {
  return (
    <HashRouter history={history}>
      <Switch>
        <Route exact path="/" component={Home} />
        <Route exact path="/event/:title/:issueNumber" component={BlogPost} />
      </Switch>
    </HashRouter>
  );
};

export default Router;
