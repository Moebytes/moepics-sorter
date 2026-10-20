<div align="left">
  <p>
    <img src="https://i.imgur.com/RjiDNHk.png" width="600"/>
  </p>
</div>

The automatic sorter aims to greatly speed up image sorting tasks on Moepictures. It will likely not be perfect, but it should significantly lessen the workload for manual sorting. Currently, it involves the following parts:

- A preferences model that determines which images to accept. The accuracy of this model is the most critical step.

- A tagger model that determines the special qualifiers to add like "comic" or "chibi". 

- A relationship model that determines the relationship qualifiers for variants, child posts, or groups.

The "alt source" feature is not implemented due to complexity but it is rarely used.

©2026 Moepictures